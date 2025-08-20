// Copyright 2023 The Casibase Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/casibase/casibase/proxy"
	aimlapi "github.com/D1m7asis/casibase-aimlapi-go"
)

// AIMLAPIModelProvider implements Casibase provider for AI/ML API.
type AIMLAPIModelProvider struct {
	subType     string
	secretKey   string
	siteName    string
	siteUrl     string
	temperature *float32
	topP        *float32
}

func NewAIMLAPIModelProvider(subType string, secretKey string, temperature float32, topP float32) (*AIMLAPIModelProvider, error) {
	p := &AIMLAPIModelProvider{
		subType:     subType,
		secretKey:   secretKey,
		siteName:    "Casibase",
		siteUrl:     "https://casibase.org",
		temperature: &temperature,
		topP:        &topP,
	}
	return p, nil
}

func (p *AIMLAPIModelProvider) GetPricing() string {
	// Pricing depends on the selected model and may change over time.
	// Please refer to AIMLAPI official pricing page.
	return `URL:
https://aimlapi.com/pricing

Notes:
- Pricing varies per model (OpenAI, Anthropic, Google, Meta, etc.)
- Always use the official page as the source of truth
`
}

// calculatePrice assigns token usage cost if known; otherwise defaults to 0 USD.
func (p *AIMLAPIModelProvider) calculatePrice(modelResult *ModelResult) error {
	var inputPricePerThousandTokens, outputPricePerThousandTokens float64

	// Example price table (incomplete, extend as needed).
	priceTable := map[string][]float64{
		// OpenAI
		"openai/gpt-4o":          {0.005, 0.015},
		"gpt-4o-2024-05-13":      {0.005, 0.015},
		"gpt-4o-mini":            {0.003, 0.006},
		"gpt-3.5-turbo":          {0.001, 0.002},

		// Anthropic
		"claude-3-5-sonnet-20240620": {0.003, 0.015},
		"claude-3-haiku-20240307":    {0.0008, 0.0024},

		// Google
		"google/gemini-2.5-pro":  {0.0025, 0.0075},
		"google/gemma-3-4b-it":   {0.0004, 0.0008},

		// Meta
		"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {0.0002, 0.0006},
		"meta-llama/Llama-3-8b-chat-hf":               {0.0002, 0.0006},

		// DeepSeek
		"deepseek-chat":     {0.0006, 0.0012},
		"deepseek-reasoner": {0.0015, 0.0030},
	}

	if priceItem, ok := priceTable[p.subType]; ok {
		inputPricePerThousandTokens = priceItem[0]
		outputPricePerThousandTokens = priceItem[1]
	} else {
		// Unknown model → fallback: free (0 USD).
		inputPricePerThousandTokens = 0
		outputPricePerThousandTokens = 0
	}

	inputPrice := getPrice(modelResult.PromptTokenCount, inputPricePerThousandTokens)
	outputPrice := getPrice(modelResult.ResponseTokenCount, outputPricePerThousandTokens)
	modelResult.TotalPrice = AddPrices(inputPrice, outputPrice)
	modelResult.Currency = "USD"
	return nil
}

func (p *AIMLAPIModelProvider) getClient() *aimlapi.Client {
	cfg, err := aimlapi.DefaultConfig(p.secretKey, p.siteName, p.siteUrl)
	if err != nil {
		panic(err)
	}
	cfg.HTTPClient = proxy.ProxyHttpClient
	return aimlapi.NewClientWithConfig(cfg)
}

func (p *AIMLAPIModelProvider) QueryText(question string, writer io.Writer, history []*RawMessage, prompt string, knowledgeMessages []*RawMessage, agentInfo *AgentInfo) (*ModelResult, error) {
	client := p.getClient()

	ctx := context.Background()
	flusher, ok := writer.(http.Flusher)
	if !ok {
		return nil, fmt.Errorf("writer does not implement http.Flusher")
	}

	model := p.subType
	if model == "" {
		// Default AIMLAPI model
		model = "openai/gpt-4o"
	}

	tokenCount, err := GetTokenSize(model, question)
	if err != nil {
		return nil, err
	}

	contextLength := getContextLength(p.subType)

	if strings.HasPrefix(question, "$CasibaseDryRun$") {
		modelResult, err := getDefaultModelResult(model, question, "")
		if err != nil {
			return nil, fmt.Errorf("cannot calculate tokens")
		}
		if contextLength > modelResult.TotalTokenCount {
			return modelResult, nil
		} else {
			return nil, fmt.Errorf("exceeds max tokens")
		}
	}

	maxTokens := contextLength - tokenCount
	if maxTokens < 0 {
		return nil, fmt.Errorf("Token count [%d] exceeds model [%s] max context [%d]", tokenCount, model, contextLength)
	}

	temperature := p.temperature
	topP := p.topP

	respStream, err := client.CreateChatCompletionStream(
		ctx,
		&aimlapi.ChatCompletionRequest{
			Model: p.subType,
			Messages: []aimlapi.ChatCompletionMessage{
				{
					Role:    aimlapi.ChatMessageRoleSystem,
					Content: "You are a helpful assistant.",
				},
				{
					Role:    aimlapi.ChatMessageRoleUser,
					Content: question,
				},
			},
			Stream:      false,
			Temperature: temperature,
			TopP:        topP,
			MaxTokens:   maxTokens,
		},
	)
	if err != nil {
		return nil, err
	}
	defer respStream.Close()

	responseStringBuilder := strings.Builder{}
	isLeadingReturn := true

	for {
		completion, streamErr := respStream.Recv()
		if streamErr != nil {
			if streamErr == io.EOF {
				break
			}
			return nil, streamErr
		}

		data := completion.Choices[0].Message.Content
		if isLeadingReturn && len(data) != 0 {
			if strings.Count(data, "\n") == len(data) {
				continue
			} else {
				isLeadingReturn = false
			}
		}

		if _, err = fmt.Fprintf(writer, "event: message\ndata: %s\n\n", data); err != nil {
			return nil, err
		}

		_, _ = responseStringBuilder.WriteString(data)
		flusher.Flush()
	}

	modelResult, err := getDefaultModelResult(p.subType, question, responseStringBuilder.String())
	if err != nil {
		return nil, err
	}

	if err := p.calculatePrice(modelResult); err != nil {
		return nil, err
	}

	return modelResult, nil
}

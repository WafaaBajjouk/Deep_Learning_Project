### For fine-tuning and predictions (eg. Llama-3.1-8B):

```python
f"""## Instruction
Given the headline and preview of a financial news article, classify the sentiment toward the provided ticker symbol. Respond only with "Positive", "Negative" or "Neutral".
* Ticker: {}
* Headline: {}
* Preview: {}
## Response
""" #newline at the end and no indentation
```

### For labeling (Llama-3.1-405B-instruct) and final comparison (eg. Llama-3.1-8B-instruct"):

```python
f"""TODO"""
```
* See [LLama 3.1 prompt formats](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1)

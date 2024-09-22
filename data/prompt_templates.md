### Labeling (Llama 3.1 instruct):

See Llama 3.1 instruct [prompt format](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1)

```python
f"""<|start_header_id|>user<|end_header_id|>
Given the headline and preview of a financial news article, classify the sentiment toward the provided ticker symbol. Respond only with "Positive", "Negative" or "Neutral".
* Ticker: {}
* Headline: {}
* Preview: {}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
""" #newline at the end and no indentation
```


### Fine-tuning (GPT 2):

```python
f"""## Instruction
Given the headline and preview of a financial news article, classify the sentiment toward the provided ticker symbol. Respond only with "Positive", "Negative" or "Neutral".
* Ticker: {}
* Headline: {}
* Preview: {}
## Response
{}""" #no indentation
```

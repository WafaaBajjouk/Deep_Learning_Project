import os
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["HUGGING_FACE_HUB_TOKEN"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

start_chunk = int(sys.argv[1])
end_chunk = int(sys.argv[2])

cache_dir = '/orfeo/fast/dssc/wbajjouk'  

print(f"Loading data and preparing tokenizer and model...")


stocks_df = pd.read_csv('stocks.csv')
crypto_df = pd.read_csv('crypto.csv')

stocks_chunks = pd.read_csv('stocks.csv', chunksize=len(stocks_df) // 5)
crypto_chunks = pd.read_csv('crypto.csv', chunksize=len(crypto_df) // 5)

tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Meta-Llama-3.1-405B-Instruct',
    token=os.environ["HUGGING_FACE_HUB_TOKEN"],
    cache_dir=cache_dir
)
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Meta-Llama-3.1-405B-Instruct',
    torch_dtype=torch.bfloat16,
    token=os.environ["HUGGING_FACE_HUB_TOKEN"],
    cache_dir=cache_dir
)

print("Tokenizer and model loaded successfully.")

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model.to('cuda')

def label_data(ticker, headline, preview):
    prompt = f"""## Instruction
Given the headline and preview of a financial news article, classify the sentiment toward the provided ticker symbol. Respond only with "Positive", "Negative" or "Neutral".
* Ticker: {ticker}
* Headline: {headline}
* Preview: {preview}
## Response
"""
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output.strip()

def process_and_label(df, ticker_col, headline_col, preview_col):
    df['label'] = df.apply(lambda x: label_data(x[ticker_col], x[headline_col], x[preview_col]), axis=1)
    return df


for i, (stocks_chunk, crypto_chunk) in enumerate(zip(stocks_chunks, crypto_chunks)):
    if start_chunk <= i < end_chunk:
        print(f"Processing chunk {i}...")

        stocks_labeled = process_and_label(stocks_chunk, 'ticker', 'headline', 'preview')
        crypto_labeled = process_and_label(crypto_chunk, 'ticker', 'headline', 'preview')

        stocks_labeled.to_csv('stocks.csv', mode='a', header=False, index=False)
        crypto_labeled.to_csv('crypto.csv', mode='a', header=False, index=False)

        print(f"Completed chunk {i}.")

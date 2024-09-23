from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForCausalLM
from huggingface_hub import login
from datasets import Dataset
from math import ceil
from time import time
import pandas as pd
import torch
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
print('GPUs:', torch.cuda.device_count(), flush=True)

def format(row):
    return {
        'text': f"""<|start_header_id|>user<|end_header_id|>
Given the headline and preview of a financial news article, classify the sentiment toward the provided ticker symbol. Respond only with "Positive", "Negative" or "Neutral".
* Ticker: {row['ticker']}
* Headline: {row['headline']}
* Preview: {row['preview']}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    }

def encode(batch):
    return tokenizer(batch['text'], return_tensors='np')

login('') 
model_path = os.path.expanduser('~/fast/models/8B')
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    pad_token='<|finetune_right_pad_id|>',
    padding_side='left',
)

dataset_files = ['dataset.csv'] 
dfs = [pd.read_csv(file) for file in dataset_files]
df = pd.concat(dfs, ignore_index=True)

dataset = Dataset.from_pandas(df).map(format).remove_columns(['ticker', 'headline', 'preview'])

start_row = 0  
end_row = len(dataset) 

dataset = dataset.select(range(start_row, end_row))
dataset = dataset.map(encode, batched=True).remove_columns(['text'])

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
    max_memory={
        0: '32GiB', 
        1: '32GiB'
    }
)
print(model.hf_device_map, flush=True)

start = time()
batch_size = 8  
data_collator = DataCollatorWithPadding(tokenizer)
generated = []

for i in range(ceil(len(dataset) / batch_size)):
    torch.cuda.empty_cache()
    print(i, end=' ', flush=True)
    batch = data_collator(dataset[i * batch_size:(i + 1) * batch_size]).to('cuda')
    generated_ids = model.generate(
        **batch,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=1
    )
    generated += tokenizer.batch_decode(generated_ids[:, -1])

print('\nElapsed:', time() - start, 's')
print(generated[-3:])

df_subset = df.iloc[start_row:end_row].copy() 
df_subset['sentiment'] = generated

output_file = f'labels/labeling_{start_row}_to_{end_row}.csv'
df_subset.to_csv(output_file, index=False)
print(f'Saved labeled data to {output_file}')

"""
TODO ACTUALLY THIS IS NOT FOR FINE-TUNING BUT FOR PREDICTING (WITH INSTRUCT MODEL)
"""

from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForCausalLM
from huggingface_hub import login
from datasets import Dataset
from math import ceil
from time import time
import pandas as pd
import torch

print('GPUs:', torch.cuda.device_count(), flush=True)

def format(row): #as instruct for simplicity (https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1)
    return {'text': f"""<|start_header_id|>user<|end_header_id|>
Given the headline and preview of a financial news article, classify the sentiment toward the provided ticker symbol. Respond only with "Positive", "Negative" or "Neutral".
* Ticker: {row['ticker']}
* Headline: {row['headline']}
* Preview: {row['preview']}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""} #newline at the end and no indentation

def encode(batch):
    return tokenizer(batch['text'], return_tensors='np') #https://huggingface.co/docs/datasets/nlp_process#map

#tokenizer
login('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') #https://huggingface.co/docs/hub/en/models-gated#download-files
model_name = 'meta-llama/Meta-Llama-3.1-8B-instruct'
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    pad_token='<|finetune_right_pad_id|>',
    padding_side='left', #https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
    cache_dir=f'/orfeo/fast/dssc/mpolo000/cache/huggingface/hub/') #download resumes by default

#dataset
df = pd.concat([pd.read_csv('data/stocks.csv'), pd.read_csv('data/crypto.csv')], ignore_index=True)
dataset = Dataset.from_pandas(df).map(format).remove_columns(['ticker', 'headline', 'preview'])
dataset = dataset.map(encode, batched=True).remove_columns(['text'])
dataset = dataset.select(range(1000)) #TODO remove after debugging

#model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto', #https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference#limits-and-further-development
    max_memory={ #https://huggingface.co/docs/accelerate/concept_guides/big_model_inference#designing-a-device-map
        0: '32GiB', #TODO tune? (may crash if too large)
        1: '32GiB'}) #TODO tune? (may crash if too large)
print(model.hf_device_map, flush=True) #https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference

#generate
start = time()
batch_size = 32 #TODO tune? (crashes if too large)
data_collator = DataCollatorWithPadding(tokenizer) #for dynamic padding applying tokenizer.pad
generated = []
for i in range(ceil(len(dataset)/batch_size)):
    print(i, end=' ', flush=True)
    batch = data_collator(dataset[i*batch_size:(i+1)*batch_size]).to('cuda')
    generated_ids = model.generate( #https://huggingface.co/docs/transformers/en/main_classes/text_generation
        **batch,
        pad_token_id=tokenizer.pad_token_id, #avoids warning
        max_new_tokens=1)
    generated = generated + tokenizer.batch_decode(generated_ids[:,-1])
print('\nElapsed:', time()-start, 's')
print(generated[-3:])

#output example (1000 news):
"""
[mpolo000@login02 deep]$ sbatch run.sh
Submitted batch job 822735
[mpolo000@login02 deep]$ cat slurm.out 
Sat Sep 21 05:03:36 CEST 2024
822735
gpu004.hpc.rd.areasciencepark.it
-----------------------------

GPUs: 2
The token has not been saved to the git credentials helper. Pass `add_to_git_credential=True` in this function directly or `--add-to-git-credential` if using via `huggingface-cli` if you want to set the git credential as well.
Token is valid (permission: read).
Your token has been saved to /u/dssc/mpolo000/.cache/huggingface/token
Login successful
Map: 100%|██████████| 182879/182879 [00:15<00:00, 12167.56 examples/s]
Map: 100%|██████████| 182879/182879 [00:31<00:00, 5725.35 examples/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:13<00:00,  3.29s/it]
{'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1, 'model.norm': 1, 'model.rotary_emb': 1, 'lm_head': 1}
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
Elapsed: 284.26036500930786 s
['Positive', 'Positive', 'Neutral']

-----------------------------
Done
Sat Sep 21 05:09:28 CEST 2024
"""

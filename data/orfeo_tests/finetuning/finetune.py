from transformers import AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorWithPadding, AutoModelForCausalLM
from huggingface_hub import login
from datasets import Dataset
from math import ceil
import pandas as pd
import torch

print('GPUs:', torch.cuda.device_count(), flush=True)

def format(row, eval=False):
    return {'text': f"""## Instruction
Given the headline and preview of a financial news article, classify the sentiment toward the provided ticker symbol. Respond only with "Positive", "Negative" or "Neutral".
* Ticker: {row['ticker']}
* Headline: {row['headline']}
* Preview: {row['preview']}
## Response
{'' if eval else row['sentiment']}"""} #no indentation

def encode(batch, tokenizer):
    return tokenizer(batch['text'], return_tensors='np') #https://huggingface.co/docs/datasets/nlp_process#map

#data collator for training
login('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') #https://huggingface.co/docs/hub/en/models-gated#download-files
model_name = 'openai-community/gpt2-xl'
train_tok = AutoTokenizer.from_pretrained(
    model_name,
    pad_token='<|endoftext|>', #the only one available by default
    padding_side='right', #https://huggingface.co/docs/transformers/model_doc/gpt2#usage-tips
    clean_up_tokenization_spaces=False, #will be default in the future for gpt2
    cache_dir='/orfeo/fast/dssc/mpolo000/cache/huggingface/hub/')
train_coll = DataCollatorForLanguageModeling(train_tok, mlm=False) #for labels and dynamic padding applying train_tok.pad

#TODO is DataCollatorForLanguageModeling the right choice? Is it equivalent to use trl.SFTTrainer instead of Trainer?

#data collator for testing
test_tok = AutoTokenizer.from_pretrained(
    model_name,
    pad_token='<|endoftext|>', #https://github.com/huggingface/transformers/pull/7552#issue-497255933
    padding_side='left', #https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
    clean_up_tokenization_spaces=False,
    cache_dir='/orfeo/fast/dssc/mpolo000/cache/huggingface/hub/')
test_coll = DataCollatorWithPadding(test_tok) #for dynamic padding applying test_tok.pad

#train set
df = pd.concat([pd.read_csv('data/stocks.csv'), pd.read_csv('data/crypto.csv')], ignore_index=True)
df['sentiment'] = 'Positive' #TODO use labeled dataset instead
train_set, test_set = Dataset.from_pandas(df).train_test_split(test_size=0.1, seed=42).values()
train_set = train_set.map(format).remove_columns(['ticker', 'headline', 'preview', 'sentiment'])
train_set = train_set.map(lambda batch: encode(batch, train_tok), batched=True).remove_columns(['text'])
train_set = train_set.select(range(100)) #TODO remove after debugging
train_set, val_set = train_set.train_test_split(test_size=0.1, seed=42).values() #TODO is val_set used?

#test set
test_set = test_set.map(lambda row: format(row, eval=True)).remove_columns(['ticker', 'headline', 'preview'])
test_set = test_set.map(lambda batch: encode(batch, test_tok), batched=True).remove_columns(['text'])
test_set = test_set.select(range(100)) #TODO remove after debugging






print(train_coll(train_set[:2])) #TODO remove after debugging

#TODO continue integrating code from below and gpt2 notebook

################################################################# ex code about labeling test

# #model
# model = AutoModelForCausalLM.from_pretrained( #TODO `torch_dtype=torch.float16`?
#     model_name,
#     device_map='auto', #https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference#limits-and-further-development
#     max_memory={ #https://huggingface.co/docs/accelerate/concept_guides/big_model_inference#designing-a-device-map
#         0: '32GiB', #TODO tune? (may crash if too large)
#         1: '32GiB'}, #TODO tune? (may crash if too large)
#     cache_dir='/orfeo/fast/dssc/mpolo000/cache/huggingface/hub/') #download resumes by default
# print(model.hf_device_map, flush=True) #https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference

# #generate
# start = time()
# batch_size = 32 #TODO tune? (crashes if too large)
# data_collator = DataCollatorWithPadding(tokenizer) #for dynamic padding applying tokenizer.pad
# generated = []
# for i in range(ceil(len(dataset)/batch_size)):
#     print(i, end=' ', flush=True)
#     batch = data_collator(dataset[i*batch_size:(i+1)*batch_size]).to('cuda')
#     generated_ids = model.generate( #https://huggingface.co/docs/transformers/en/main_classes/text_generation
#         **batch,
#         pad_token_id=tokenizer.pad_token_id, #avoids warning
#         max_new_tokens=1)
#     generated = generated + tokenizer.batch_decode(generated_ids[:,-1])
# print('\nElapsed:', time()-start, 's')
# print(generated[-3:])


from transformers import AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorWithPadding, AutoModelForCausalLM, TrainingArguments, Trainer
from huggingface_hub import login
from datasets import Dataset
from math import ceil
import pandas as pd
import torch

print('GPUs:', torch.cuda.device_count(), flush=True)

def format(row, eval):
    return {'text': f"""## Instruction
Given the headline and preview of a financial news article, classify the sentiment toward the provided ticker symbol. Respond only with "Positive", "Negative" or "Neutral".
* Ticker: {row['ticker']}
* Headline: {row['headline']}
* Preview: {row['preview']}
## Response
{'' if eval else row['sentiment']}"""} #no indentation

def encode(batch, eval):
    if eval:
        return test_tok(batch['text'], return_tensors='np') #https://huggingface.co/docs/datasets/nlp_process#map
    return train_tok(
        batch['text'],
        return_tensors='np',
        return_special_tokens_mask=True) #https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForLanguageModeling

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

#data collator for testing
test_tok = AutoTokenizer.from_pretrained(
    model_name,
    pad_token='<|endoftext|>', #https://github.com/huggingface/transformers/pull/7552#issue-497255933
    padding_side='left', #https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
    clean_up_tokenization_spaces=False,
    cache_dir='/orfeo/fast/dssc/mpolo000/cache/huggingface/hub/')
test_coll = DataCollatorWithPadding(test_tok) #for dynamic padding applying test_tok.pad

#train set
df = pd.read_csv('balanced_labeling_data.csv')
train_set, test_set = Dataset.from_pandas(df).train_test_split(test_size=0.1, seed=42).values()
train_set = train_set.map(lambda row: format(row, eval=False)).remove_columns(['ticker', 'headline', 'preview', 'sentiment'])
train_set = train_set.map(lambda batch: encode(batch, eval=False), batched=True).remove_columns(['text'])
train_set = train_set.filter(lambda row: len(row['input_ids']) < train_tok.model_max_length)
train_set = train_set.select(range(1000)) #TODO remove after debugging
train_set, val_set = train_set.train_test_split(test_size=0.1, seed=42).values()

#test set
test_set = test_set.map(lambda row: format(row, eval=True)).remove_columns(['ticker', 'headline', 'preview'])
test_set = test_set.map(lambda batch: encode(batch, eval=True), batched=True).remove_columns(['text'])
max_new_tokens = 2 #Positive: [21604, 1800], Negative: [32863, 876], Neutral: [8199, 6815]
test_set = test_set.filter(lambda row: len(row['input_ids']) < test_tok.model_max_length-max_new_tokens)
test_set = test_set.select(range(1000)) #TODO remove after debugging

# print(train_coll([train_set[i] for i in range(2)]), flush=True) #helpful to understand

#model
model = AutoModelForCausalLM.from_pretrained( #`torch_dtype=torch.float16`?
    model_name,
    device_map='auto', #https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference#limits-and-further-development
    max_memory={ #https://huggingface.co/docs/accelerate/concept_guides/big_model_inference#designing-a-device-map
        0: '32GiB', #may crash if too large
        1: '32GiB'}, #may crash if too large
    cache_dir='/orfeo/fast/dssc/mpolo000/cache/huggingface/hub/') #download resumes by default
print(model.hf_device_map, flush=True) #https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference

#training
args = TrainingArguments(
    output_dir='tmp_trainer', #`pip install tensorboard` to also save logs
    overwrite_output_dir=False, #to continue training (manually delete dir to restart)
    eval_strategy='steps',
    per_device_train_batch_size=16, #crashes if too large (ok if auto_find_batch_size=True)
    per_device_eval_batch_size=16, #crashes if too large (ok if auto_find_batch_size=True)
    torch_empty_cache_steps=None, #default None
    learning_rate=5e-5, #default 5e-5
    num_train_epochs=2.0, #increase to continue a training that ended correctly
    logging_steps=100, #also sets eval_steps to same value by default
    save_steps=1000, #must be a round multiple of eval_steps
    save_total_limit=2, #still retains best checkpoint if load_best_model_at_end=True
    load_best_model_at_end=True,
    group_by_length=True, #why not
    auto_find_batch_size=True) #keeps training with lower batch size if crashes
trainer = Trainer(
    model=model,
    args=args,
    data_collator=train_coll,
    train_dataset=train_set,
    eval_dataset=val_set)
try:
    trainer.train(resume_from_checkpoint=True) #see logs with eg. `ssh -L 9999:localhost:6006 mpolo000@195.14.102.215` + `tensorboard --logdir .`
except ValueError:
    trainer.train()


#TODO use or delete code below
# #generate
# batch_size = 8 #TODO tune?
# generated = []
# for i in range(ceil(len(test_set)/batch_size)):
#     print(i, end=' ', flush=True)
#     batch = test_coll(test_set[i*batch_size:(i+1)*batch_size]).to('cuda')
#     generated_ids = model.generate( #https://huggingface.co/docs/transformers/en/main_classes/text_generation
#         **batch,
#         pad_token_id=test_tok.pad_token_id, #avoids warning
#         max_new_tokens=max_new_tokens)
#     generated = generated + test_tok.batch_decode(generated_ids[:,-1])
# print(generated[-3:])

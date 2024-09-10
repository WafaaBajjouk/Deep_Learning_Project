from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

login(token="")

model_id = "meta-llama/Meta-Llama-3.1-8B"

def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",  
        offload_folder="./offload", 
        offload_state_dict=True  
    )

    model.config.use_cache = False

    return model, tokenizer

model, tokenizer = get_model_and_tokenizer(model_id)

dataset = load_dataset('csv', data_files='data/formatted_sentiment_dataset.csv')

print(dataset)

lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

sft_config = SFTConfig(output_dir='financial-sentiment-llama3.1-8B', max_seq_length=512)

training_arguments = TrainingArguments(
    output_dir='financial-sentiment-llama3.1-8B',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=3,
    max_steps=250,
    fp16=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    args=training_arguments,
    tokenizer=tokenizer,
    peft_config=lora_config,
    dataset_text_field="text",  
    packing=False
)

trainer.train()

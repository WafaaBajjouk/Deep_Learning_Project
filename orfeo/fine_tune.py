from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from evaluate import load
from peft import LoraConfig, get_peft_model

# Load Financial PhraseBank dataset with trust_remote_code=True
dataset = load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)

# Preprocess the dataset
def preprocess_data(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=29)

encoded_dataset = dataset.map(preprocess_data, batched=True)

# Split the dataset into training and validation sets
train_test_split = encoded_dataset['train'].train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

# Load accuracy metric with trust_remote_code=True
metric = load('accuracy', trust_remote_code=True)

# Function to compute metrics
def compute_metrics(p):
    predictions, labels = p
    preds = predictions.argmax(-1)
    return metric.compute(predictions=preds, references=labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    save_total_limit=1,
    remove_unused_columns=False,
    logging_strategy="steps",
    log_level="info",
    report_to="none"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

# Save the model
model.save_pretrained('./finetuned_finbert')
tokenizer.save_pretrained('./finetuned_finbert')

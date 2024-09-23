import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import logging
import time

logging.basicConfig(filename='download8B_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def download_and_save_model():
    model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'  
    save_directory = os.path.expanduser('~/fast/models/8B')

    os.makedirs(save_directory, exist_ok=True)

    login()

    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=save_directory)
    model.save_pretrained(save_directory)
    logging.info(f"Model download and save time: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_directory)
    tokenizer.save_pretrained(save_directory)
    logging.info(f"Tokenizer download and save time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    download_and_save_model()

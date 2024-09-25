# Random useful things that won't be in presentation

## Conda

1. From login node:
    1. [Install](https://orfeo-doc.areasciencepark.it/HPC/python-environment/) on `scratch` from login node
        * Say "yes" at the end of the installation and ignore doc above from "If you select to not auto-activate conda..."
    2. Restart the shell ("base" env activates automatically)
    3. `conda deactivate` and `conda config --set auto_activate_base false` to avoid "base" to activate automatically
    4. From login node, `conda create -n deep-epyc`
    5. Enter a GPU node with `srun` with enough RAM, CPUs and time
2. From GPU node:
    1. `conda activate deep-epyc`
    2. Inside your env:
        1. `conda install pandas`
        2.  ```bash
            pip install transformers
            pip install torch
            pip isntall datasets
            pip install accelerate
            pip install tensorboard
            pip install trl
            ```

# For Fine-tuning part of presentation

---

## Fine-tuning GPT-2

### 1. Starting point

---

### Dataset obtained

![](pics/labeled_dataset.png)
* **Columns**: `ticker`, `headline`, `preview`, `sentiment`
* ~111,000+ **rows**
* **Balanced** w.r.t. sentiment

---

### Prompt template

![](pics/prompt_template.png)
* **Instruction tuning** format
* **Markdown** to structure text

---

### Out of the box predictions

![](pics/out_of_the_box_pred.png)
* Trained to **continue text**

![](pics/useless_helper.png)
* Introduction to **help** but useless

---

## Fine-tuning GPT-2

### 2. Data preprocessing

---

### Train-test-validation split

* **Train**: ~98,000 news (~88%)
* **Test**: ~11,000 news (~10%)
* **Validation**: ~2000 news (~2%)

---

### Tokenization

1. Breaks down prompt into **smaller units** (tokens)
2. Maps tokens to **numbers** in [0, 50,256]
    * Same used by **OpenAI** for training
3. **Discards** results longer than 1024 tokens (~20 news)
* **Examples:**
    * 'Given the' -> `[15056, 262]`
    * 'Positive' -> `[21604, 1800]`

---

### TODO









---

More precisely, inputs are sequences of continuous text of a certain length and the targets are the same sequence, shifted one token (word or piece of word) to the right. The model uses internally a mask-mechanism to make sure the predictions for the token i only uses the inputs from 1 to i but not the future tokens.



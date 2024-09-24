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
---

# For fine-tuning part of presentation

TODO

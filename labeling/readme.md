### Files

1. **`labeling.job`**: SLURM job script to submit an array of jobs for processing data chunks.
2. **`labeling.py`**: py script that performs sentiment analysis on chunks of data using the Meta-LLaMA model.


### Python Script (`labeling.py`)

- **Process**:
  - **Data Loading**: Reads data from `stocks.csv` and `crypto.csv`.
  - **Model Preparation**: Loads the tokenizer and model from Hugging Face, specifying a cache directory (`/orfeo/fast/dssc/wbajjouk`).
  - **Chunk Processing**: Processes each data chunk in parallel, classifies sentiments, and appends the results to the original files.

###  Job Script (`labeling.job`)

- **Configuration**:
  - **`--job-name=labeling`**: Sets the job name.
  - **`--partition=GPU`**: Specifies the GPU partition.
  - **`--gres=gpu:2`**: Requests 2 GPUs per task.
  - **`--nodes=2`**: Requests 2 nodes.
  - **`--ntasks-per-node=2`**: Sets 2 tasks per node.
  - **`--cpus-per-task=8`**: Allocates 8 CPU cores per task.
  - **`--time=02:00:00`**: Sets the maximum runtime to 2 hours.
  - **`--output=labeling.out`** and **`--error=labeling.err`**: Redirects all output and error messages to single files.
  - **`--array=0-4`**: Creates a job array of 5 tasks, where each task processes a different chunk of data.



### How to Run

1. **Prepare the Environment**:
   - Ensure you have access to ORFEO .
   - Make sure you have the `transformers` library and the Hugging Face model dependencies installed.

2. **Submit the Job**:
   - Save the SLURM script (`labeling.job`) and Python script (`labeling.py`) in the same directory.
   - Submit the job to SLURM using the following command:
     ```bash
     sbatch labeling.job
     ```

3. **Monitor the Job**:
   - Check the job status with:
     ```bash
     squeue -u your_username
     ```
   - Review the output (`labeling.out`) and error (`labeling.err`) files for progress and debugging.

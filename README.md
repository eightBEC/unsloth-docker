# Unsloth Docker

A Dockerfile for Unsloth.ai

## Intro

Training LLMs is oftern limited by the available VRAM of your GPU or other resources like time. [Unsloth](https://github.com/unslothai/unsloth) is a great library that helps you train LLMs faster and with less memory. Based on their [benchmarks](https://github.com/unslothai/unsloth?tab=readme-ov-file#-performance-benchmarking) up to 2x faster and with up to 80% less memory.

The following examples shows a minimum **training code example** and a **Dockerfile** you can use in your environment to get started training your models faster.

## Prerequisites

- Docker / Podman

### Example: Minimum Trainer Code for Unsloth

The following example shows how to fine-tune your model with Unsloth. The code is put together based on the examples provided on [Unsloth Github](https://github.com/unslothai/unsloth).

```Python
# file: unsloth_trainer.py
import os
import torch

from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_from_disk
from unsloth import FastLanguageModel

MODEL_ID = "unsloth/gemma-7b-bnb-4bit" # Quantized models from unsloth for faster downloading
TRAINING_DATA_PATH = "path/to/training-dataset"
OUTPUT_DATA_PATH = "path/where/model/is/stored"
NUM_EPOCHS = 1

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=2048, # adjust to your sequence length
    dtype=None,
    load_in_4bit=True,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=1133,
    use_rslora=False,
    loftq_config=None,
)

dataset = ds.load_from_disk(TRAINING_DATA_PATH)

sft_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=data_collator,
    formatting_func=format_prompts_func,
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False, 
    args=TrainingArguments(
        gradient_accumulation_steps=4,
        auto_find_batch_size=True,
        warmup_steps=5,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=2.5e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=1133,
        output_dir=OUTPUT_DATA_PATH,
    ),
)
sft_trainer.train()

try:
    model.save_pretrained_merged(
        os.path.join(OUTPUT_DATA_PATH, "model-16bit"),
        tokenizer,
        save_method="merged_16bit",
    )
except Exception as e:
    print("Error saving merged_16bit model")
    print(e)

try:
    # Merge to 4bit
    model.save_pretrained_merged(
        os.path.join(OUTPUT_DATA_PATH, "model-4bit"),
        tokenizer,
        save_method="merged_4bit",
    )
except Exception as e:
    print("Error saving merged_4bit model")
    print(e)


try:
    # Just LoRA adapters
    model.save_pretrained_merged(
        os.path.join(OUTPUT_DATA_PATH, "model-lora"),
        tokenizer,
        save_method="lora",
    )
except Exception as e:
    print("Error saving lora model")
    print(e)

```

### Example: Dockerfile for Unsloth

This Dockerfile uses the NVIDIA CUDA base image to provide a stable foundation. If you wanto to run this on Red Hat OpenShift please remember to add the non-priveleged user accordingly.

```Dockerfile
# Start from the NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set a fixed model cache directory.
ENV TORCH_HOME=/root/.cache/torch

# Install Python and necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential python3.10 python3-pip python3.10-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Update pip and setuptools
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

# Install PyTorch with CUDA 12.1 support and other essential packages
# Use a dedicated conda env 
RUN conda create --name unsloth_env python=3.10
RUN echo "source activate unsloth_env" > ~/.bashrc
ENV PATH /opt/conda/envs/unsloth_env/bin:$PATH

# As described in the Unsloth.ai Github
RUN conda install -n unsloth_env -y pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install matplotlib
RUN pip install --no-deps trl peft accelerate bitsandbytes
RUN pip install autoawq

# Copy the fine-tuning script into the container
COPY ./unsloth_trainer.py /trainer/unsloth.trainer.py

WORKDIR /trainer

# endless running task to avoid container to be stopped
CMD [ "/bin/sh" , "-c", "tail -f /dev/null" ]
```

## Further information

The sample python and Dockerfiles can also be found on my [Github](https://github.com/eightBEC/unsloth-docker/tree/main).
For those of you interested in diving deeper, please refer to the [Unsloth Github](https://github.com/unslothai/unsloth) for the latest updates, models, etc. - This library is developing balzingly fast.

#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/cogvideox-2b"
    --model_name "cogvideox-t2v"  # ["cogvideox-t2v"]
    --model_type "t2v"
    --training_type "lora"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "finetune_output"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_root "/data/zihao/one_data"
    --caption_column "prompt.txt"
    --video_column "videos.txt"
    --train_resolution "49x480x720"  # (frames x height x width), frames should be 8N+1
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 1000 # number of training epochs
    --seed 42 # random seed
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 10 # save checkpoint every x steps
    --checkpointing_limit 2 # maximum number of checkpoints to keep, after which the oldest one is deleted
    --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation false  # ["true", "false"]
    --validation_dir "/absolute/path/to/your/validation_set"
    --validation_steps 20  # should be multiple of checkpointing_steps
    --validation_prompts "prompts.txt"
    --gen_fps 16
)
export MAIN_PROCESS_PORT=12354
export MASTER_PORT=12354
# Combine all arguments and launch training
accelerate launch --main_process_port=12354 train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"

## Release Notes

This repository contains code for pre-training, finetuning, retrieving and evaluating the retrieval-augmented models. Our codebase is developed and modified from the original [ATLAS](https://github.com/facebookresearch/atlas) codebase. We annotate the modifed code with the prefix `custom_`. Please refer to the original codebase to setup the [ATLAS](https://github.com/facebookresearch/atlas) environment.

## Directory structure

Training and finetuning scripts: ./ (Main directory)\
Evaluation scripts: ./evaluation_scripts/\
Helping functions: ./src/\
preprocessing functions: ./preprocessing/


## Usage

### Model Pretraining

```bash
PASSAGE_FILES="sample_train.jsonl"
TRAIN_FILES="sample_train.jsonl"
EVAL_FILES="sample_test.jsonl"
MODEL_PATH=".."
SAVE_DIR="./experiments/"
PASSAGE_DIRNAME=atlas-mlm-gen-S2ROC-topicwise
EXPERIMENT_NAME=atlas-mlm-gen-Atlantic-220M-topicwise
TRAIN_STEPS=10000

srun python custom_train_textwithstruct.py \
    --shuffle \
    --train_retriever \
    --gold_score_mode pdist \
    --use_gradient_checkpoint_reader \
    --use_gradient_checkpoint_retriever \
    --precision fp32 \
    --shard_optim \
    --shard_grads \
    --temperature_gold 0.01 \
    --temperature_score 0.01 \
    --query_side_retriever_training \
    --target_maxlength 16 \
    --reader_model_type google/t5-${SIZE}-lm-adapt \
    --dropout 0.1 \
    --weight_decay 0.01 \
    --lr 4e-5 \
    --lr_retriever 4e-5 \
    --scheduler linear \
    --text_maxlength 512 \
    --model_path ${MODEL_PATH} \
    --train_data ${TRAIN_FILES} \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 2 \
    --n_context 20 \
    --retriever_n_context 20 \
    --retriever_n_context_beforererank 50  \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --eval_freq 11000 \
    --log_freq 4 \
    --total_steps ${TRAIN_STEPS} \
    --warmup_steps 5 \
    --save_freq 250 \
    --main_port $port \
    --write_results \
    --task qa \
    --index_mode flat \
    --save_index_n_shards 128 \
    --passages ${PASSAGE_FILES} \
    --save_index_path ${SAVE_DIR}/${EXPERIMENT_NAME}/saved_index_with_struct_emb/ \
    --load_subset_textindex \
    --subset_textindex_name Physics,Chemistry,Materials-Science \
    --load_index_path ${SAVE_DIR}/${PASSAGE_DIRNAME}/saved_index/ \
    --passage_path ${SAVE_DIR}/${PASSAGE_DIRNAME}/saved_index_with_struct_emb/ \
    --projector
```

### Model Finetuning

```bash
PASSAGE_FILES="sample_train.jsonl"
TRAIN_FILES="sample_train.jsonl"
EVAL_FILES="sample_test.jsonl"
MODEL_PATH=".."
SAVE_DIR="./experiments/"
PASSAGE_DIRNAME=atlas-mlm-gen-S2ROC-topicwise
EXPERIMENT_NAME=atlas-mlm-gen-Atlantic-220M-nuclearQA-insttun
TRAIN_STEPS=100

srun python custom_train_textwithstruct.py \
    --shuffle \
    --train_retriever \
    --gold_score_mode pdist \
    --use_gradient_checkpoint_reader \
    --use_gradient_checkpoint_retriever \
    --precision fp32 \
    --shard_optim \
    --shard_grads \
    --temperature_gold 0.01 \
    --temperature_score 0.01 \
    --query_side_retriever_training \
    --target_maxlength 16 \
    --reader_model_type google/t5-${SIZE}-lm-adapt \
    --dropout 0.1 \
    --weight_decay 0.01 \
    --lr 4e-5 \
    --lr_retriever 4e-5 \
    --scheduler linear \
    --text_maxlength 512 \
    --model_path ${MODEL_PATH} \
    --train_data ${TRAIN_FILES} \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 2 \
    --n_context 20 \
    --retriever_n_context 20 \
    --retriever_n_context_beforererank 50  \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --eval_freq 11000 \
    --log_freq 4 \
    --total_steps ${TRAIN_STEPS} \
    --warmup_steps 5 \
    --save_freq 250 \
    --main_port $port \
    --write_results \
    --task qa \
    --index_mode flat \
    --save_index_n_shards 128 \
    --passages ${PASSAGE_FILES} \
    --save_index_path ${SAVE_DIR}/${EXPERIMENT_NAME}/saved_index_with_struct_emb/ \
    --load_subset_textindex \
    --subset_textindex_name Physics,Chemistry,Materials-Science \
    --load_index_path ${SAVE_DIR}/${PASSAGE_DIRNAME}/saved_index/ \
    --passage_path ${SAVE_DIR}/${PASSAGE_DIRNAME}/saved_index_with_struct_emb/ \
    --projector
```


### NuclearQA Evaluation

```bash
MODEL_TO_EVAL=".."
EVAL_FILES="./QA_queries_sample.jsonl"
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=atlas-mlm-gen-Atlantic-220M_evalnuclearQA
PASSAGE_DIRNAME=atlas-mlm-gen-S2ROC-topicwise
PRECISION="fp32" # "bf16"

srun python ./evaluation_scripts/custom_evaluate_textwithstruct.py \
    --name ${EXPERIMENT_NAME} \
    --generation_max_length 100 \
    --target_maxlength 100 \
    --gold_score_mode "pdist" \
    --precision ${PRECISION} \
    --reader_model_type google/t5-${size}-lm-adapt \
    --text_maxlength 512 \
    --model_path ${MODEL_TO_EVAL} \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 2 \
    --n_context 20 \
    --retriever_n_context 20 \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port $port \
    --index_mode "flat"  \
    --task qa \
    --write_results \
    --load_subset_textindex \
    --subset_textindex_name Physics,Chemistry,Mathematics \
    --load_index_path ${PASSAGE_DIRNAME}/saved_index/ \
    --passage_path ${PASSAGE_DIRNAME}/saved_index_with_struct_emb/ \
    --projector
```

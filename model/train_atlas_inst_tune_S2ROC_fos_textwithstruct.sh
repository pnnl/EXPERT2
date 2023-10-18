export HF_HOME="/cache"
export TORCH_EXTENSIONS_DIR="/cache"

# python make_hostfile.py

echo $CUDA_VISIBLE_DEVICES

# assumes 4 nodes, each with 8 GPUs
DATA_DIR=/test_atlas/
SIZE=base # lets use large, (slower than base, but still quite fast and accessible, but less accurate than xl or xxl)

port=$(shuf -i 15000-16000 -n 1)
PASSAGE_FILES="/Engg.jsonl"

TRAIN_FILES="/instructions_sample_train.jsonl"

EVAL_FILES="/instructions_sample.jsonl"

MODEL_PATH="/atlas-mlm-gen-S2ROC-220M-struct-domainwise/checkpoint_11thround_42000/step-1000"

SAVE_DIR="/experiments"
PASSAGE_DIRNAME=atlas-mlm-gen-S2ROC-topicwise
EXPERIMENT_NAME=atlas-mlm-gen-Atlantic-220M-fos-instun
TRAIN_STEPS=10500

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
    --task base \
    --index_mode flat \
    --save_index_n_shards 128 \
    --passages ${PASSAGE_FILES} \
    --save_index_path ${SAVE_DIR}/${EXPERIMENT_NAME}/saved_index_with_struct_emb/ \
    --load_subset_textindex \
    --subset_textindex_name Physics,Chemistry,Bio-1,Art,Political-Science,Economics,Geology,Computer-Science,Environmental-Science,Materials-Science,Engineering \
    --load_index_path ${SAVE_DIR}/${PASSAGE_DIRNAME}/saved_index/ \
    --passage_path ${SAVE_DIR}/${PASSAGE_DIRNAME}/saved_index_with_struct_emb/ \
    --projector
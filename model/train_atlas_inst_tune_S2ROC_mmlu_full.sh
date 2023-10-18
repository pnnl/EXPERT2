export HF_HOME="/cache"
export TORCH_EXTENSIONS_DIR="/cache"

# python make_hostfile.py

echo $CUDA_VISIBLE_DEVICES

# assumes 4 nodes, each with 8 GPUs
DATA_DIR=/test_atlas
SIZE=base # lets use large, (slower than base, but still quite fast and accessible, but less accurate than xl or xxl)

port=$(shuf -i 15000-16000 -n 1)
PASSAGE_FILES="/Engg.jsonl"

TRAIN_FILES="${DATA_DIR}/data/mmlu_data/full/train.jsonl"

EVAL_FILES="${DATA_DIR}/data/mmlu_data/full/combined_valid.jsonl ${DATA_DIR}/data/mmlu_data/full/combined_test.jsonl"

MODEL_PATH="/models/atlas/base/"

SAVE_DIR="/experiments"
PASSAGE_DIRNAME=atlas-mlm-gen-S2ROC-topicwise
EXPERIMENT_NAME=atlas-mlm-gen-Atlas-220M-mmlufull-finetune
TRAIN_STEPS=2000

srun python custom_train.py \
    --shuffle \
    --train_retriever \
    --gold_score_mode ppmean \
    --use_gradient_checkpoint_reader \
    --use_gradient_checkpoint_retriever \
    --precision fp32 \
    --shard_optim \
    --shard_grads \
    --temperature_gold 0.01 \
    --temperature_score 0.01 \
    --query_side_retriever_training \
    --target_maxlength 16 \
    --refresh_index 100000000000 \
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
    --n_context 30 \
    --retriever_n_context 30 \
    --retriever_n_context_beforererank 60  \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --eval_freq 2000 \
    --log_freq 4 \
    --total_steps ${TRAIN_STEPS} \
    --warmup_steps 50 \
    --save_freq 250 \
    --main_port $port \
    --write_results \
    --task multiple_choice \
    --index_mode flat \
    --save_index_n_shards 128 \
    --passages ${PASSAGE_FILES} \
    --save_index_path ${SAVE_DIR}/${EXPERIMENT_NAME}/saved_index/ \
    --load_subset_textindex \
    --subset_textindex_name Physics,Chemistry,Mathematics,Bio-1,Med-1,Art,History,Sociology,Philosophy,Business,Economics,Computer-Science \
    --load_index_path ${SAVE_DIR}/${PASSAGE_DIRNAME}/saved_index/ \
    --passage_path ${SAVE_DIR}/${PASSAGE_DIRNAME}/saved_index/ \
    --multiple_choice_train_permutations all \
    --multiple_choice_eval_permutations cyclic \




    
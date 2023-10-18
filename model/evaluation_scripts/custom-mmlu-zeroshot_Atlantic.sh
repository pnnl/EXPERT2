export HF_HOME="/cache"
export TORCH_EXTENSIONS_DIR="/cache"

echo $CUDA_VISIBLE_DEVICES

size=base
DATA_DIR='/test_atlas'

port=$(shuf -i 15000-16000 -n 1)

MODEL_PATH="/atlas-mlm-gen-Atlantic-220M-mmlufull-finetune/checkpoint/step-2000/"

EVAL_FILES="${DATA_DIR}/data/mmlu_data/full/combined_test.jsonl"

SAVE_DIR=${DATA_DIR}/experiments_textwithstruct
EXPERIMENT_NAME=atlas-mlm-gen-Atlantic-220M-mmlufull-fulltestevalv3
PASSAGE_DIRNAME=${DATA_DIR}/experiments/atlas-mlm-gen-S2ROC-topicwise
PRECISION="fp32" # "bf16"

srun python custom_evaluate_textwithstruct.py \
    --precision ${PRECISION} \
    --target_maxlength 100 \
    --reader_model_type google/t5-${size}-lm-adapt \
    --text_maxlength 512 \
    --model_path ${MODEL_PATH} \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 2 \
    --n_context 20 \
    --retriever_n_context 20 \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port $port \
    --write_results \
    --task multiple_choice \
    --multiple_choice_train_permutations all \
    --multiple_choice_eval_permutations cyclic \
    --index_mode flat \
    --load_index_path ${PASSAGE_DIRNAME}/saved_index/ \
    --passage_path ${PASSAGE_DIRNAME}/saved_index_with_struct_emb/ \
    --load_subset_textindex \
    --subset_textindex_name Physics,Chemistry,Mathematics,Bio-1,Med-1,Art,History,Sociology,Philosophy,Business,Economics,Computer-Science \
    --projector \
 

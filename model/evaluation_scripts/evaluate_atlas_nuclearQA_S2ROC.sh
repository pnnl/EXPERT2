export TORCH_EXTENSIONS_DIR="/cache"

echo $CUDA_VISIBLE_DEVICES

DATA_DIR=/test_atlas/
size=base

YEAR=${1:-"2017"}

MODEL_TO_EVAL="./atlas-mlm-gen-S2ROC-220M-nq-fewshot/checkpoint/step-30/"

port=$(shuf -i 15000-16000 -n 1)

EVAL_FILES="/nuclearQA/QA_queries_sample.jsonl"

SAVE_DIR=${DATA_DIR}/experiments
EXPERIMENT_NAME=atlas-mlm-gen-S2ROC-220M-nq-instun-nuclearQAeval-step30v2
PASSAGE_DIRNAME=${DATA_DIR}/experiments/atlas-mlm-gen-S2ROC-topicwise
PRECISION="fp32" # "bf16"

srun python custom_evaluate_textwithstruct.py \
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

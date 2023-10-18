export TORCH_EXTENSIONS_DIR="/cache"

echo $CUDA_VISIBLE_DEVICES

DATA_DIR=/test_atlas/
size=base

YEAR=${1:-"2017"}

MODEL_TO_EVAL="/experiments_textwithstruct/atlas-mlm-gen-Atlantic-220M-fos-instun/checkpoint/step-10500/"
MODEL_FOR_INDEX="/models/atlas/base/"

port=$(shuf -i 15000-16000 -n 1)

EVAL_FILES="/mag/instructions_sample_test.jsonl"

SAVE_DIR=${DATA_DIR}/experiments
EXPERIMENT_NAME=atlas-mlm-gen-Atlantic-220M-fos-instun-mageval-step10kv3
PASSAGE_DIRNAME=${DATA_DIR}/experiments/atlas-mlm-gen-S2ROC-topicwise
PRECISION="fp32" # "bf16"

srun python custom_evaluate_adapretr_dynamic_textwithstruct.py \
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
    --per_gpu_batch_size_domainindex 600 \
    --no_sel_indices 3 \
    --n_context 20 \
    --retriever_n_context 20 \
    --checkpoint_dir ${DATA_DIR}/experiments_textwithstruct/ \
    --main_port $port \
    --index_mode "flat"  \
    --task base \
    --write_results \
    --load_subset_textindex \
    --load_index_path ${PASSAGE_DIRNAME}/saved_index/ \
    --passage_path ${PASSAGE_DIRNAME}/saved_index_with_struct_emb/ \
    --index_model_path ${MODEL_FOR_INDEX} \
    --projector
    # --retrieve_only 
    # --qa_prompt_format "{question}"
    # --subset_textindex_name Biology,Medicine-1 \

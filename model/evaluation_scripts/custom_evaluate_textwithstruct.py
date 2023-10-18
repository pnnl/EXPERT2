# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.cuda
import torch.distributed as dist

from src import dist_utils, slurm, util
from src.index_io import load_or_initialize_index, save_embeddings_and_index
from src.model_io_textwithstruct import create_checkpoint_directories, load_or_initialize_atlas_model
from src.options import get_options
from src.tasks import get_task

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def _get_eval_data_iterator(opt, data_path, task):
    data_iterator = task.data_iterator(
        data_path, opt.global_rank, opt.world_size, opt=opt, is_eval=True)
    data_iterator = filter(None, map(task.process, data_iterator))
    data_iterator = list(task.batch_iterator(
        data_iterator, opt.per_gpu_batch_size))

    if dist.is_initialized():
        len_data = torch.tensor(
            len(data_iterator), device=torch.device("cuda"))
        dist.all_reduce(len_data, torch.distributed.ReduceOp.MAX)
        dist.barrier()
        if len(data_iterator) < len_data.item():
            data_iterator.extend(
                [{} for _ in range(len_data.item() - len(data_iterator))])

    return data_iterator


@torch.no_grad()
def run_retrieval_only(model, index, opt, data_path, step=None):
    model.eval()
    metrics = defaultdict(lambda: [])
    dataset_wpred = []
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    task = get_task(opt, reader_tokenizer)
    data_iterator = _get_eval_data_iterator(opt, data_path, task)

    for i, batch in enumerate(data_iterator):
        query = batch.get("query", [""])
        answers = batch.get("target", [""])
        batch_metadata = batch.get("metadata")
        query_enc = model.retriever_tokenize(query)
        retrieved_passages, _, _ = unwrapped_model.retrieve(
            index,
            opt.n_context,
            query,
            query_enc["input_ids"].cuda(),
            query_enc["attention_mask"].cuda(),
            batch_metadata=batch_metadata,
            filtering_fun=task.filter,
        )
        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0):
            continue
        for k in range(len(retrieved_passages)):
            if opt.write_results:
                gold = [
                    answers[k]] if not "answers" in batch else batch["answers"][k]
                ex = {"query": query[k], "answers": gold,
                      "passages": retrieved_passages[k]}
                if batch_metadata is not None:
                    ex["metadata"] = batch_metadata[k]
                if "id" in batch:
                    ex["id"] = batch["id"][k]
                dataset_wpred.append(ex)

    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)

    return metrics


@torch.no_grad()

def evaluate(model, index, opt, data_path, step=None):
    model.eval()
    metrics = defaultdict(lambda: [])
    dataset_wpred = []
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    task = get_task(opt, reader_tokenizer)
    data_iterator = _get_eval_data_iterator(opt, data_path, task)

    for i, batch in enumerate(data_iterator):
        query = batch.get("query", [""])
        answers = batch.get("target", [""])
        batch_metadata = batch.get("metadata")
        target_tokens = batch.get("target_tokens")
        query_enc, labels, decoder_input_ids = unwrapped_model.tokenize(
            query, answers, target_tokens=target_tokens)

        if not opt.use_file_passages:
            query_ids_retriever = query_enc["input_ids"].cuda()
            query_mask_retriever = query_enc["attention_mask"].cuda()

            retrieved_passages, passages_score, query_emb_fordevice = unwrapped_model.retrieve(
                index,
                opt.n_context,
                query,
                query_ids_retriever,
                query_mask_retriever,
                batch_metadata=batch_metadata,
                filtering_fun=task.filter,
            )
            # print("##passage eval", len(retrieved_passages[0]))

        else:
            assert "passages" in batch, "cant use use_file_passages mode without passing in passages"
            retrieved_passages = [p[: opt.n_context]
                                  for p in batch["passages"]]

        ## edit by sai
        ## get bacth struct embedding on retrieved passages
        batch_struct_list = []
        for ele2 in retrieved_passages:               
            batch_struct_list.append(torch.Tensor(np.array([ele1['struct_emb'] for ele1 in ele2])))

        batch_struct_emb_tensor = torch.cat(batch_struct_list, axis=0)
        batch_struct_emb = batch_struct_emb_tensor.to(query_emb_fordevice.device)

        # print("##", len(query))
        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0):
            continue

        # print("## retrieved passage", len(retrieved_passages[0]), retrieved_passages[0][0])

        reader_tokens, _ = unwrapped_model.tokenize_passages(
            query, retrieved_passages)

        ## edit by sai
        ## reshape batch struct emb
        n_context_training = min(opt.n_context, reader_tokens["input_ids"].size(1))

        batch_struct_emb_tensor = batch_struct_emb_tensor.view(len(batch_struct_list),batch_struct_list[0].shape[0],-1)
        # print(batch_struct_emb_tensor.shape)
        batch_struct_emb_tensor_training = batch_struct_emb_tensor[:, :n_context_training].contiguous()
        # print(batch_struct_emb_tensor_training.shape)
        batch_struct_emb_tensor_training = batch_struct_emb_tensor_training.view(batch_struct_emb_tensor_training.shape[0]*batch_struct_emb_tensor_training.shape[1],batch_struct_emb_tensor_training.shape[2])
        # print(batch_struct_emb_tensor_training.shape)
        batch_struct_emb_tensor_training = batch_struct_emb_tensor_training.to(query_emb_fordevice.device)

        ## PROJECTION LAYER from struct emb into text emb space
        batch_struct_emb_training = unwrapped_model.projector_output(batch_struct_emb_tensor_training)

        batch_struct_emb_training = torch.reshape(batch_struct_emb_training, (batch_struct_emb_training.shape[0], 1, batch_struct_emb_training.shape[1]))

        if "eval_loss" in task.metrics:
            eval_loss, logits = unwrapped_model.compute_reader_loss_and_logits(
                reader_tokens, decoder_input_ids, labels, batch_struct_emb_training)
            metrics["eval_loss"].append(eval_loss)

        # logger.info(f"eval loss/logits comp done:{batch_struct_emb.shape, len(query)}")
        

        generation, extra_generation = unwrapped_model.generate(
            reader_tokens, query, choices=batch["choices"] if "choices" in batch else None , 
            batch_struct_emb=batch_struct_emb_training,
        )

        probs = torch.stack(extra_generation.scores, dim=1).softmax(-1)
        val_probs, ind = torch.max(probs, dim=-1)

        for k, g in enumerate(generation):
            if opt.decoder_prompt_format is not None:
                query_ids = reader_tokenizer.encode(
                    opt.decoder_prompt_format.format_map({"query": query[k]}), add_special_tokens=False
                )
                g = g[len(query_ids) + 1:]
            pred = reader_tokenizer.decode(g, skip_special_tokens=True)
            gold = [answers[k]] if not "answers" in batch else batch["answers"][k]
            sample_metrics = task.evaluation(pred, gold)
            for key, value in sample_metrics.items():
                metrics[key].append(value)

            if opt.write_results:
                ex = {"query": query[k], "answers": gold, "generation": pred,
                      "logits":val_probs[5*k:5*k+5,:].cpu().data.numpy().tolist()}
                if not opt.dont_write_passages:
                    ex["passages"] = retrieved_passages[k]
                    ex["passages_scores"] = passages_score[k]
                if batch_metadata is not None:
                    ex["metadata"] = batch_metadata[k]
                if opt.task == "multiple_choice":
                    ex["choice_logits"] = task.get_choice_logits(logits[k])
                if "id" in batch:
                    ex["id"] = batch["id"][k]
                for ele in ex["passages"]:
                    # logger.info(f"passage ele:{ele }")
                    ele["struct_emb"]=""
                    
                ex["sequences_id"] = extra_generation.sequences[3*k:3*k+3,:].cpu().data.numpy().tolist()
                ex["sequences_text"] = [reader_tokenizer.decode(extra_generation.sequences[ind,:], skip_special_tokens=True) for ind in range(3*k,3*k+3)]

                dataset_wpred.append(ex)

        # evaluating for total_steps no of queries
        # if i > opt.total_steps:
        #     break

    metrics, dataset_wpred = task.evaluation_postprocessing(
        metrics, dataset_wpred)
    # logger.info(f"data wpred done:{metrics, dataset_wpred}")
    metrics = util.avg_dist_dict(task.metrics, metrics)
    metrics = {key: value if key == "eval_loss" else 100 *
               value for key, value in metrics.items()}
    
    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)

    return metrics

if __name__ == "__main__":
    options = get_options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)

    logger = util.init_logger(
        opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log"))
    if opt.is_main:
        options.print_options(opt)

    logger.info(f"world size: {dist_utils.get_world_size()}")
    
    index, passages = load_or_initialize_index(opt)

    model, _, _, _, _, opt, step, _, _,_,_ = load_or_initialize_atlas_model(
        opt, eval_only=True)
    
    logger.info("Start Evaluation")

    dist_utils.barrier()

    if not opt.use_file_passages and opt.load_index_path is None:
        indexing_start = time.time()
        logger.info("building index")

        model.build_index(
            index, passages, opt.per_gpu_embedder_batch_size, logger)

        if opt.save_index_path is not None:
            save_embeddings_and_index(index, opt)

    for data_path in opt.eval_data:
        dataset_name = os.path.basename(data_path)
        logger.info(f"Start Evaluation on {data_path}")
        if opt.retrieve_only:
            run_retrieval_only(model, index, opt, data_path, step)
        else:
            metrics = evaluate(model, index, opt, data_path, step)
            log_message = f"Dataset: {dataset_name}"
            for k, v in metrics.items():
                log_message += f" | {v:.3f} {k}"
            logger.info(log_message)

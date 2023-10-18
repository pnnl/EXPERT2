# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import defaultdict
from scipy.stats import entropy
import numpy as np
import torch
import torch.cuda
import torch.distributed as dist

from src import dist_utils, slurm, util
from src.index_io import load_or_initialize_index, save_embeddings_and_index
# from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model, load_or_initialize_atlas_model_forindex
from src.options import get_options
from src.tasks import get_task
import sys
import linecache

"""
This file created by Sai Munikoti

This python file mofifies evaluate.py in terms of 
(1) logging logits 
(2) evaluating total input queries w/o restriction  

"""


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
def get_entropy(prob_list, min_val=0.7, max_val=3):

    '''
    
    This function returns predictive entropy score, normalized over a user-defined or pre-defined range.
    We use the Shanon Entropy to calcualte the predictive entropy for a given set of generated sequences.

    Input:
        min_val (dtype: float): The minimum value for normalization over range
        max_val (dtype: float): The maximum value for normalization over range
    
    Output:
        ent_val: Range-Normlized Predictive Entropy score for generated sequences
    
    Citation:
        Shannon, Claude Elwood. "A mathematical theory of communication." 
        ACM SIGMOBILE mobile computing and communications review 5.1 (2001): 3-55.
    
    '''

    # Calculating Mean Predictive Entropy
    ent_val = entropy(prob_list)

    # Range-Normalization
    ent_val = (ent_val - min_val)/(max_val - min_val)
    
    # Checks to keep score between 0 and 1
    if ent_val < 0:
        ent_val = ent_val * (-1)

    if ent_val > 1:
        ent_val = 1

    return ent_val
    

@torch.no_grad()
def evaluate(model, index, opt, data_path, step=None):
    # tracing line by line

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

            retrieved_passages, passages_score, _ = unwrapped_model.retrieve(
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

        # print("##", len(query))
        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0):
            continue

        # print("## retrieved passage", len(retrieved_passages[0]), retrieved_passages[0][0])

        reader_tokens, _ = unwrapped_model.tokenize_passages(
            query, retrieved_passages)

        if "eval_loss" in task.metrics:
            eval_loss, logits = unwrapped_model.compute_reader_loss_and_logits(
                reader_tokens, decoder_input_ids, labels)
            metrics["eval_loss"].append(eval_loss)

        generation, extra_generation = unwrapped_model.generate(
            reader_tokens, query, choices=batch["choices"] if "choices" in batch else None
        )
        
        # logger.info(f"extra generation 2:{ len(query), len(extra_generation),len(extra_generation.scores),extra_generation.scores[0].shape,len(extra_generation.sequences),extra_generation.sequences[0].shape,extra_generation.sequences[0]}")
        
        logits_sequencewise = torch.stack(extra_generation.scores, dim=1)
        probs = torch.stack(extra_generation.scores, dim=1).softmax(-1)

        val_probs, ind = torch.max(probs, dim=-1)

        # logger.info(f"extra generation 1:{extra_generation.scores}")
        # logger.info(f"extra generation 2:{extra_generation.sequences}")
        # logger.info(f"val probs:{len(val_probs[0].cpu().data.numpy().tolist()),len(val_probs[1].cpu().data.numpy().tolist())}")

        
        # logger.info(f"prob size:{probs.shape,val_probs.shape,val_probs[0],logits_sequencewise.shape}")
        
        # logger.info(f"prob sum:{[torch.sum(probs[0,ele,:]) for ele in range(0,probs.shape[1])]}")

        # gen_sequences = extra_generation.sequences[:, len(reader_tokens['input_ids']):]
        # logger.info(f"gen seq size:{gen_sequences.size()}")
        

        for k, g in enumerate(generation):

            if opt.decoder_prompt_format is not None:
                query_ids = reader_tokenizer.encode(
                    opt.decoder_prompt_format.format_map({"query": query[k]}), add_special_tokens=False
                )
                g = g[len(query_ids) + 1:]

            pred = reader_tokenizer.decode(g, skip_special_tokens=True)
            # pred_all = reader_tokenizer.decode(g, skip_special_tokens=False)
            # logger.info(f"gen pred:{g, pred}")
            
            gold = [answers[k]] if not "answers" in batch else batch["answers"][k]
            sample_metrics = task.evaluation(pred, gold)
            for key, value in sample_metrics.items():
                metrics[key].append(value)

            ## number of beams in generation is 5
            if opt.write_results:
                # logits_list = val_probs[k].cpu().data.numpy().tolist()
                ex = {"query": query[k], "answers": gold, "generation": pred, 
                      "logits":val_probs[5*k:5*k+5,:].cpu().data.numpy().tolist(),
                    #   "logits_sequencewise":logits_sequencewise[5*k:5*k+5,:].cpu().data.numpy().tolist()
                      }
                    #   "entropy":get_entropy(logits_list),
                    #   "norm_entropy":get_entropy(logits_list)}
                # ex = {"query": query[k], "answers": gold, "generation": pred}
                if not opt.dont_write_passages:
                    ex["passages"] = retrieved_passages[k]
                    ex["passages_scores"] = passages_score[k]
                if batch_metadata is not None:
                    ex["metadata"] = batch_metadata[k]
                if opt.task == "multiple_choice":
                    ex["choice_logits"] = task.get_choice_logits(logits[k])
                if "id" in batch:
                    ex["id"] = batch["id"][k]
                # ex["scores_dic"] = [ele.cpu().data.numpy().tolist() for ele in extra_generation.scores]
                ex["sequences_id"] = extra_generation.sequences[3*k:3*k+3,:].cpu().data.numpy().tolist()
                
                ex["sequences_text"] = [reader_tokenizer.decode(extra_generation.sequences[ind,:], skip_special_tokens=True) for ind in range(3*k,3*k+3)]

                # ex["sequences_id"] = extra_generation.sequences[3*k:3*k+3,:].cpu().data.numpy().tolist()

                dataset_wpred.append(ex)

        # logger.info(f"dataset_wpred :{dataset_wpred[0]}")
        # exit()
        # evaluating for total_steps no of queries
        # if i > opt.total_steps:
        #     break

    
    metrics, dataset_wpred = task.evaluation_postprocessing(
        metrics, dataset_wpred)

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

    model, _, _, _, _, opt, step, _, _, = load_or_initialize_atlas_model(
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

    def traceit(frame, event, arg):
        if event == "line":
            lineno = frame.f_lineno
            try:
                filename = frame.f_globals["__file__"]
                if filename == "<stdin>":
                    filename = "traceit.py"
                if (filename.endswith(".pyc") or
                    filename.endswith(".pyo")):
                    filename = filename[:-1]
                name = frame.f_globals["__name__"]
                line = linecache.getline(filename, lineno)
                if ("src.modeling_t5" in name) or ("src.fid" in name) or ("src.atlas" in name):
                    print("%s:%s: %s" % (name, lineno, line.rstrip()))
                # else:
                #     print("%s:%s: %s" % (name, lineno, line.rstrip()))
            except:
                print("--")
        return traceit

    for data_path in opt.eval_data:
        dataset_name = os.path.basename(data_path)
        logger.info(f"Start Evaluation on {data_path}")
        if opt.retrieve_only:
            run_retrieval_only(model, index, opt, data_path, step)
        else:
            # sys.settrace(traceit)
            metrics = evaluate(model, index, opt, data_path, step)
            log_message = f"Dataset: {dataset_name}"
            for k, v in metrics.items():
                log_message += f" | {v:.3f} {k}"
            logger.info(log_message)


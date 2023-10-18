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
from pathlib import Path

from src import dist_utils, slurm, util
from src.index_io import load_or_initialize_index, save_embeddings_and_index, load_subset_index
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model, load_or_initialize_atlas_model_forindex
from src.options import get_options
from src.tasks import get_task
import pandas as pd

"""
This file created by Sai Munikoti
This python file mofifies custom_evaluate.py in terms of :
(1) This file group the input query into different subject topics and then adaptively load relevant to each group
and then change index as group changes

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

def _get_eval_data_iterator_forindexsel(opt, data_path, task):
    data_iterator = task.data_iterator(
        data_path, opt.global_rank, opt.world_size, opt=opt, is_eval=True)
    data_iterator = filter(None, map(task.process, data_iterator))
    ## edit by sai
    # data_iterator = list(task.batch_iterator(
    #     data_iterator, opt.per_gpu_batch_size))
    data_iterator = list(task.batch_iterator(
        data_iterator, opt.per_gpu_batch_size_domainindex))

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
        retrieved_passages, _ = unwrapped_model.retrieve(
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
    
    # print("## query started")

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

        # print("## retrieved passage", len(retrieved_passages[0]))

        reader_tokens, _ = unwrapped_model.tokenize_passages(
            query, retrieved_passages)

        if "eval_loss" in task.metrics:
            eval_loss, logits = unwrapped_model.compute_reader_loss_and_logits(
                reader_tokens, decoder_input_ids, labels)
            metrics["eval_loss"].append(eval_loss)

        # generation = unwrapped_model.generate(
        #     reader_tokens, query, choices=batch["choices"] if "choices" in batch else None
        # )
        ## edit by sai
        ## getting logits from prediction

        generation, extra_generation = unwrapped_model.generate(
            reader_tokens, query, choices=batch["choices"] if "choices" in batch else None
        )

        # logits_sequencewise = torch.stack(extra_generation.scores, dim=1)

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
                    #   "logits_sequencewise":logits_sequencewise[5*k:5*k+5,:].cpu().data.numpy().tolist()}                
                if not opt.dont_write_passages:
                    ex["passages"] = retrieved_passages[k]
                    ex["passages_scores"] = passages_score[k]
                if batch_metadata is not None:
                    ex["metadata"] = batch_metadata[k]
                if opt.task == "multiple_choice":
                    ex["choice_logits"] = task.get_choice_logits(logits[k])
                if "id" in batch:
                    ex["id"] = batch["id"][k]

                ex["sequences_id"] = extra_generation.sequences[3*k:3*k+3,:].cpu().data.numpy().tolist()
                # print("## sequences_id done",extra_generation.sequences[3*k,:],g)
                # print("## sequences text done",reader_tokenizer.decode(extra_generation.sequences[3*k,:], skip_special_tokens=True))

                ex["sequences_text"] = [reader_tokenizer.decode(extra_generation.sequences[ind,:], skip_special_tokens=True) for ind in range(3*k,3*k+3)]
                # ex["sequences_text"] = extra_generation.sequences[3*k:3*k+3,:].cpu().data.numpy().tolist() 
                # print("## sequences text done 2 ")
                dataset_wpred.append(ex)

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

## edit by sai
## function to select domain index from the batch query
@torch.no_grad()
def get_domainindex(model, opt, data_path, domain_wise_index_vec, domain_map_names):
    model.eval()
    metrics = defaultdict(lambda: [])
    dataset_wpred = []
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    task = get_task(opt, reader_tokenizer)
    data_iterator = _get_eval_data_iterator_forindexsel(opt, data_path, task)

    for i, batch in enumerate(data_iterator):
        query = batch.get("query", [""])
        logger.info(f"len query 0: {len(query),len(query[0])}")
        # get only input
        query = [ele.split("###")[3] for ele in query]
        # logger.info(f"query shape:{len(query),query[0]}")
        answers = batch.get("target", [""])
        # batch_metadata = batch.get("metadata")
        target_tokens = batch.get("target_tokens")
        query_enc, labels, decoder_input_ids = unwrapped_model.tokenize(
            query, answers, target_tokens=target_tokens)

        query_ids_retriever = query_enc["input_ids"].cuda()
        query_mask_retriever = query_enc["attention_mask"].cuda()

        logger.info(f"len query: {len(query),len(query[0])}")
        # logger.info(f"query: {query[0]}")
        # logger.info(f"domain vec shape: {domain_wise_index_vec.shape}")
        ## get batch query embedding vectors

        selected_domain_name = unwrapped_model.get_topdomain_index_name(
            query,
            query_ids_retriever,
            query_mask_retriever,
            domain_wise_index_vec,
            domain_map_names,
            opt.no_sel_indices
        )

        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0):
            continue
            

    selected_domain_name = ",".join(selected_domain_name)

    return selected_domain_name

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

    # index, passages = load_or_initialize_index(opt)

    # model, _, _, _, _, opt, step, _, _ = load_or_initialize_atlas_model(
    #     opt, eval_only=True)

    # load domain representing vector
    domain_map_names = {
    0: "Art",
    1: "Geography",
    2: "History",
    3: "Sociology",
    4: "Philosophy",
    5: "Political-Science",
    6: "Psychology",
    7: "Business",
    8: "Economics",
    9: "Geology",
    10: "Bio-1",
    11: "Mathematics",
    12: "Computer-Science",
    13: "Environmental-Science",
    14: "Engineering",
    15: "Materials-Science",
    16: "Bio-2",
    17: "Physics",
    18: "Chemistry",
    19: "Med-1",
    20: "Med-2",
    21: "Med-3",
    22: "Med-4"
    }

    domain_list = ['Art','History', 'Sociology', 
                   'Philosophy', 'Political-Science','Psychology', 
                    'Geology', 'Mathematics','Engineering','Economics',
                    'Computer-Science', 'Environmental-Science','Business',
                    'Materials-Science', 'Biology', 'Physics','Chemistry',
                    'Medicine','Materials science','Political science','Law','Linguistics']
    
    # domain_map_names = {
    # 0: "HSS",
    # 1: "Medicine",
    # 2: "Engg",
    # 3: "PhyScience"
    # }

    domain_vector_list = [torch.load(opt.load_index_path+domain_map_names[ele]+"/mean_emb.pt") for ele in range(len(domain_map_names))]
    domain_wise_index_vec = torch.cat(domain_vector_list, dim=1)
    logger.info(f"domain_wise_index_vec mean {torch.mean(domain_wise_index_vec,dim=0)}")

    ## edit by sai
    ### load model and index once and evaluate on different files
    # eval_files_list = opt.eval_data.split(",")
    
    for eval_file in opt.eval_data:
        logger.info(f"eval file {eval_file}")
        # step = 0
        if not opt.use_file_passages and opt.load_index_path is None:
            indexing_start = time.time()
            logger.info("building index")
            model.build_index(
                index, passages, opt.per_gpu_embedder_batch_size, logger)

            if opt.save_index_path is not None:
                save_embeddings_and_index(index, opt)

        ## edit by sai ##
        ## load data and group by topic ##
        df = pd.read_json(eval_file, lines=True)

        df = df.loc[df['target'].isin(domain_list)]
        grouped = df.groupby(['target'])


        # df['label_type']= df['target'].apply(lambda x : len(x.split(',')))
        # grouped = df.loc[df['label_type']==1].groupby('target')

        logger.info(f"groupby size: {grouped.size(), len(grouped)}")

        ## load models
        index_model, _, _, _, _, opt, step, _, _ = load_or_initialize_atlas_model_forindex(opt, eval_only=True)
        model, _, _, _, _, opt, step, _, _ = load_or_initialize_atlas_model(opt, eval_only=True)

        logger.info("Start Evaluation")

        dir_path = Path(opt.checkpoint_dir) / opt.name

        ## save temp file and itearte over each topic group
        for name,group in grouped:
            data_path = os.path.dirname(eval_file)+"/instructions_sample_" + name[0] + ".jsonl"
            
            if not os.path.exists(data_path):
                group.to_json(data_path, orient='records', lines=True)

            dataset_name, _ = os.path.splitext(os.path.basename(data_path))
            dataset_name = f"{dataset_name}-step-{step}"
            final_path = dir_path / f"{dataset_name}.jsonl"

            if os.path.exists(final_path):
                logger.info(f"final path exists: {final_path}")
                continue

            # for data_path in [eval_file]: #TODO change loop variable for multiple eval files
            dataset_name = os.path.basename(data_path) 
            logger.info(f"Start Evaluation on {data_path}")
            try:
                if opt.retrieve_only:
                    run_retrieval_only(model, index, opt, data_path, step)
                else:
                    ## edit by sai
                    # initialize atlas (retriever with bert base uncased)
                    # index_model, _, _, _, _, opt, step, _, _ = load_or_initialize_atlas_model_forindex(opt, eval_only=True)
                    # model, _, _, _, _, opt, step, _, _ = inixt_atlas_model(opt, eval_only=True)
                    ## select domain as per the query batch 
                    selected_domain_name = get_domainindex(index_model, opt, data_path, domain_wise_index_vec, domain_map_names)
                    logger.info(f"sel domain index {selected_domain_name}")
                    
                    ## load selected index 
                    index, passages = load_subset_index(opt, selected_domain_name)
                
                    # dist_utils.barrier()

                    # model, _, _, _, _, opt, step, _, _ = load_or_initialize_atlas_model(opt, eval_only=True)        
        
                    metrics = evaluate(model, index, opt, data_path, step)

                    log_message = f"Dataset: {dataset_name}"
                    for k, v in metrics.items():
                        log_message += f" | {v:.3f} {k}"
                    logger.info(log_message)

                    ## release index and passages from memory
                    del index, passages
                    torch.cuda.empty_cache()
            except:
                logger.info(f"break Evaluation on {data_path}")
                continue
            
            

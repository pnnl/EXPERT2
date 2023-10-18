# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging

from src import dist_utils
from src.index import DistributedFAISSIndex, DistributedIndex

logger = logging.getLogger(__name__)


def load_passages(filenames, maxload=-1):
    def process_jsonl(
        fname,
        counter,
        passages,
        world_size,
        global_rank,
        maxload,
    ):
        def load_item(line):
            if line.strip() != "":
                item = json.loads(line)
                assert "id" in item
                if "title" in item and "section" in item and len(item["section"]) > 0:
                    item["title"] = f"{item['title']}: {item['section']}"
                return item
            else:
                print("empty line")

        for line in open(fname):
            if maxload > -1 and counter >= maxload:
                break

            ex = None
            if (counter % world_size) == global_rank:
                ex = load_item(line)
                passages.append(ex)
            counter += 1
        return passages, counter

    counter = 0
    passages = []
    global_rank = dist_utils.get_rank()
    world_size = dist_utils.get_world_size()
    for filename in filenames:

        passages, counter = process_jsonl(
            filename,
            counter,
            passages,
            world_size,
            global_rank,
            maxload,
        )

    return passages


def save_embeddings_and_index(index, opt: argparse.Namespace) -> None:
    """
    Saves embeddings and passages files. It also saves faiss index files if FAISS mode is used.
    """
    index.save_index(opt.save_index_path,
                     opt.save_index_n_shards)  # opt.save_index_n_shards=128


def load_or_initialize_index(opt):
    if opt.index_mode == "flat":
        index = DistributedIndex()
    elif opt.index_mode == "faiss":
        index = DistributedFAISSIndex(
            opt.faiss_index_type, opt.faiss_code_size)
    else:
        raise ValueError(f"unsupported index mode {opt.index_mode}")

    if opt.load_index_path is not None:
        logger.info(
            f"Loading index from: {opt.load_index_path} with index mode: {opt.index_mode}")
        if opt.index_mode == "faiss":
            logger.info(
                f"loading faiss index type {opt.faiss_index_type} with parameters {opt.faiss_code_size}")
        ## edit by sai
        logger.info(f"subset textindex name:{ opt.subset_textindex_name}")
        index.load_index(opt.load_index_path, opt.save_index_n_shards, opt.load_subset_textindex, opt.subset_textindex_name)
        passages = [index.doc_map[i] for i in range(len(index.doc_map))]
        # print("## loaded index", len(passages))
        logger.info(f"loaded index: {len(passages)}" )
    else:
        logger.info(f"Loading passages from: {opt.passages}")
        passages = []
        if not opt.use_file_passages:
            passages = load_passages(opt.passages, opt.max_passages)
            logger.info(f"initializing embeddings: {opt.passages}" )
            index.init_embeddings(passages)

    return index, passages


## load a subset of index 
def load_subset_index(opt, subset_index_names):
    '''
    subset_index_names: string of domain names seprated by comma
    '''
    if opt.index_mode == "flat":
        index = DistributedIndex()
    elif opt.index_mode == "faiss":
        index = DistributedFAISSIndex(
            opt.faiss_index_type, opt.faiss_code_size)
    else:
        raise ValueError(f"unsupported index mode {opt.index_mode}")

    if opt.load_index_path is not None:
        logger.info(
            f"Loading index from: {opt.load_index_path} with index mode: {opt.index_mode}")
        if opt.index_mode == "faiss":
            logger.info(
                f"loading faiss index type {opt.faiss_index_type} with parameters {opt.faiss_code_size}")
        ## edit by sai
        index.load_index(opt.load_index_path, opt.save_index_n_shards, opt.load_subset_textindex, subset_index_names)
        passages = [index.doc_map[i] for i in range(len(index.doc_map))]
        # print("## loaded index", len(passages))
        logger.info(f"loaded index: {len(passages)}" )
    else:
        logger.info(f"Loading passages from: {opt.passages}")
        passages = []
        if not opt.use_file_passages:
            passages = load_passages(opt.passages, opt.max_passages)
            logger.info(f"initializing embeddings: {opt.passages}" )
            index.init_embeddings(passages)

    return index, passages


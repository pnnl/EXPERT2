# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import errno
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
import transformers

import src.fid
from src import dist_utils
from src.atlas import Atlas
from src.retrievers import Contriever, DualEncoderRetriever, UntiedDualEncoderRetriever
from src.util import cast_to_precision, set_dropout, set_optim
from src.retrievers import GAT
from torch import nn

Number = Union[float, int]

logger = logging.getLogger(__name__)


def get_checkpoint_path(opt):
    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    return checkpoint_path


def create_checkpoint_directories(opt):
    checkpoint_path = get_checkpoint_path(opt)
    os.makedirs(checkpoint_path, exist_ok=True)
    if opt.save_index_path:
        os.makedirs(opt.save_index_path, exist_ok=True)
    dist_utils.barrier()
    return checkpoint_path, opt.save_index_path


def load_retriever(opt, opt_checkpoint=None):
    if opt.use_file_passages:
        return None, None

    contriever_encoder = Contriever.from_pretrained(opt.retriever_model_path)
    retriever_tokenizer = transformers.AutoTokenizer.from_pretrained(
        opt.retriever_model_path)

    # once you have done query side training you cannot go back to a parameter-tied retriever
    if opt_checkpoint is not None:
        retriever_is_untied = opt_checkpoint.query_side_retriever_training or opt.query_side_retriever_training
    else:
        retriever_is_untied = opt.query_side_retriever_training

    if retriever_is_untied:
        retriever = UntiedDualEncoderRetriever(opt, contriever_encoder)
    else:
        retriever = DualEncoderRetriever(opt, contriever_encoder)

    return retriever, retriever_tokenizer


def load_retriever_graph(opt, opt_checkpoint=None):

    gat = GAT(opt.embeddings_dim, opt.node_hidemb_size, opt.embeddings_dim)

    return gat


def _convert_state_dict_from_dual_encoder_retriever(state_dict):
    """handles when we want to load an UntiedDualEncoderRetriever from a DualEncoderRetriever state dict"""
    new_state_dict = {}
    for k, tensor in state_dict.items():
        if k.startswith("retriever"):
            new_state_dict[k.replace(
                "retriever.contriever", "retriever.passage_contriever")] = tensor
            new_state_dict[k.replace(
                "retriever.contriever", "retriever.query_contriever")] = tensor
        else:
            new_state_dict[k] = tensor
    return new_state_dict


def load_reader(opt):
    reader = None
    if not opt.retrieve_only:
        reader = src.fid.FiD.from_pretrained(opt.reader_model_type)
        # logger.info(f"cross attn stats{opt.compute_crossattention_stats}")
        if opt.compute_crossattention_stats or "eval" in opt.gold_score_mode or "std" in opt.gold_score_mode:
            logger.info(f"cross attn stats{opt.compute_crossattention_stats}")
            reader.overwrite_forward_crossattention()
            reader.create_crossattention_storage()

    reader_tokenizer = transformers.AutoTokenizer.from_pretrained(
        opt.reader_model_type)
    return reader, reader_tokenizer


def _set_reader_encoder_cfg(model, opt):
    if model.reader is not None:
        cfg = model.reader.encoder.config
        cfg.n_context = opt.n_context
        cfg.bsz = opt.per_gpu_batch_size


def _cast_atlas_to_precision(atlas_model, precision):
    if atlas_model.reader is not None:
        atlas_model.reader = cast_to_precision(atlas_model.reader, precision)
    if atlas_model.retriever is not None and precision == "bf16":
        atlas_model.retriever = cast_to_precision(
            atlas_model.retriever, precision)


def _cast_and_set_attrs_and_send_to_device(model, opt):
    _set_reader_encoder_cfg(model, opt)
    set_dropout(model, opt.dropout)
    _cast_atlas_to_precision(model, opt.precision)
    model = model.to(opt.device)
    return model


def _load_atlas_model_state(opt, opt_checkpoint, model, model_dict):

    model_dict = {
        k.replace("retriever.module", "retriever").replace("reader.module", "reader"): v for k, v in model_dict.items()
    }
    ## edit by sai
    ## load all modules
    # if opt.projector:
    #     model_dict = {
    #         k.replace("retriever.module", "retriever").replace("reader.module", "reader").replace("projector.module", "projector"): v for k, v in model_dict.items()
    #     }

    if opt.retrieve_with_rerank_bygraph:
        model_dict = {
            k.replace("retriever.module", "retriever").replace("reader.module", "reader").replace("retriever_graph.module", "retriever_graph"): v for k, v in model_dict.items()
        }

    if opt.query_side_retriever_training and not opt_checkpoint.query_side_retriever_training:
        model_dict = _convert_state_dict_from_dual_encoder_retriever(
            model_dict)

    if opt.retrieve_only:  # dont load reader if in retrieve only mode
        model_dict = {k: v for k, v in model_dict.items()
                      if not k.startswith("reader")}

    if opt.use_file_passages:  # dont load retriever if in use_file_passages mode
        model_dict = {k: v for k, v in model_dict.items()
                      if not k.startswith("retriever")}
    ## edited by sai
    current_model_dict = model.state_dict()
    current_model_dict.update(model_dict)

    # print("## copying state dict")
    # model.load_state_dict(model_dict)
    model.load_state_dict(current_model_dict)
    logger.info("## copied state dict")
    model = _cast_and_set_attrs_and_send_to_device(model, opt)
    return model


def load_atlas_model(dir_path, opt, reset_params=False, eval_only=False):
    epoch_path = os.path.realpath(dir_path)
    save_path = os.path.join(epoch_path, "model.pth.tar")
    logger.info(f"Loading {epoch_path}")
    logger.info(f"loading checkpoint {save_path}")
    checkpoint = torch.load(save_path, map_location="cpu")
    opt_checkpoint = checkpoint["opt"]
    step = checkpoint["step"]
    model_dict = checkpoint["model"]

    reader, reader_tokenizer = load_reader(opt)
    retriever, retriever_tokenizer = load_retriever(opt, opt_checkpoint)

    if opt.retrieve_with_rerank_bygraph:
        # initalize GNN model for reranking passages
        retriever_graph = load_retriever_graph(opt)
    else:
        retriever_graph = None

    # if opt.projector:
    #     ## edit by sai
    #     ## projects from struct emb to text emb space
    #     projector = ProjectorNetwork(input_dim=64, output_dim=768)
    # else:
    #     projector = None

    model = Atlas(opt, reader, retriever,
                  reader_tokenizer, retriever_tokenizer, retriever_graph)

    # TODO
    model = _load_atlas_model_state(opt, opt_checkpoint, model, model_dict)

    if eval_only:
        return model, None, None, None, None, opt_checkpoint, step, None, None

    if not reset_params:
        logger.info(f"reset_params")
        optimizer, scheduler, retr_optimizer, retr_scheduler, retr_graph_optimizer, retr_graph_scheduler,proj_optimizer, proj_scheduler = set_optim(
            opt_checkpoint, model)

        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        optimizer, scheduler, retr_optimizer, retr_scheduler, retr_graph_optimizer, retr_graph_scheduler, proj_optimizer, proj_scheduler = set_optim(
            opt, model)

    return model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt_checkpoint, step, retr_graph_optimizer, retr_graph_scheduler


def init_atlas_model(opt, eval_only):
    reader, reader_tokenizer = load_reader(opt)
    retriever, retriever_tokenizer = load_retriever(opt)

    logger.info("initialize atlas model")

    # if opt.projector:
    #     ## edit by sai
    #     ## projects from struct emb to text emb space
    #     projector = ProjectorNetwork(input_dim=64, output_dim=768)
    # else:
    #     projector = None

    if opt.retrieve_with_rerank_bygraph:
        # initalize GNN model for reranking passages
        retriever_graph = load_retriever_graph(opt)
    else:
        retriever_graph = None

    model = Atlas(opt, reader, retriever,
                  reader_tokenizer, retriever_tokenizer, retriever_graph)

    model = _cast_and_set_attrs_and_send_to_device(model, opt)

    if eval_only:
        return model, None, None, None, None, opt, 0, None, None

    optimizer, scheduler, retr_optimizer, retr_scheduler, retr_graph_optimizer, retr_graph_scheduler, proj_optimizer, proj_scheduler = set_optim(
        opt, model)
    return model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt, 0, retr_graph_optimizer, retr_graph_scheduler

## load atlas model for index search
def load_or_initialize_atlas_model_forindex(opt, eval_only=False):

    if opt.index_model_path == "none":
        # if not os.path.exists(latest_checkpoint_path):  # Fresh run: initialize model
            return init_atlas_model(opt, eval_only)
        # else:  # Resume continue training from last ckpt
        #     load_path, reset_params = latest_checkpoint_path, False
    else:  # fresh finetune run, initialized from old model
        load_path, reset_params = opt.index_model_path, True

    model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt_checkpoint, loaded_step, \
        retr_graph_optimizer, retr_graph_scheduler = load_atlas_model(
            load_path, opt, reset_params=reset_params, eval_only=eval_only
        )
    logger.info(f"Model loaded from {load_path}")
    step = 0 if opt.index_model_path != "none" else loaded_step

    return model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt, step, retr_graph_optimizer, retr_graph_scheduler    

def load_or_initialize_atlas_model(opt, eval_only=False):
    """
    Either initializes a Atlas from t5 and contriever or loads one from disk.

    if opt.model_path is "none" and {opt.checkpoint_dir/opt.name} doesn't exist, it will init a Atlas

    or, if opt.model_path is "none" and {opt.checkpoint_dir/opt.name} does exist, it will load the Atlas at opt.checkpoint_dir/opt.name/latest

    or, if opt.model_path is not "none" it will load the saved Atlas in opt.model_path
    """
    checkpoint_path = get_checkpoint_path(opt)
    latest_checkpoint_path = os.path.join(
        checkpoint_path, "checkpoint", "latest")

    if opt.model_path == "none":
        # return init_atlas_model(opt, eval_only)
        if not os.path.exists(latest_checkpoint_path):  # Fresh run: initialize model
            return init_atlas_model(opt, eval_only)
        else:  # Resume continue training from last ckpt
            load_path, reset_params = latest_checkpoint_path, False
    else:  # fresh finetune run, initialized from old model
        load_path, reset_params = opt.model_path, True

    model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt_checkpoint, loaded_step, \
        retr_graph_optimizer, retr_graph_scheduler = load_atlas_model(
            load_path, opt, reset_params=reset_params, eval_only=eval_only
        )
    logger.info(f"Model loaded from {load_path}")
    step = 0 if opt.model_path != "none" else loaded_step

    return model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt, step, retr_graph_optimizer, retr_graph_scheduler

def save_atlas_model(model, optimizer, scheduler, retr_optimizer, retr_scheduler, step, opt, dir_path, name,
                     retr_graph_optimizer, retr_graph_scheduler):
    def symlink_force(target, link_name):
        try:
            os.symlink(target, link_name)
        except OSError as e:
            if e.errno == errno.EEXIST:
                os.remove(link_name)
                os.symlink(target, link_name)
            else:
                raise e

    model_to_save = model.module if hasattr(model, "module") else model
    path = os.path.join(dir_path, "checkpoint")
    epoch_path = os.path.join(path, name)  # "step-%s" % step)
    os.makedirs(epoch_path, exist_ok=True)
    cp = os.path.join(path, "latest")
    fp = os.path.join(epoch_path, "model.pth.tar")
    if opt.save_optimizer and opt.shard_optim:
        optimizer.consolidate_state_dict()
        # proj_optimizer.consolidate_state_dict()
        if retr_optimizer:
            retr_optimizer.consolidate_state_dict()
            retr_graph_optimizer.consolidate_state_dict()
    optim_state = optimizer.state_dict() if opt.save_optimizer else None
    # proj_optim_state = proj_optimizer.state_dict() if opt.save_optimizer else None

    if retr_optimizer and opt.save_optimizer:
        retr_optim_state = retr_optimizer.state_dict()
        retr_graph_optim_state = retr_graph_optimizer.state_dict()
        print("# optim state is available")
    else:
        retr_optim_state = None
        retr_graph_optim_state = None


    checkpoint = {
        "step": step,
        "model": model_to_save.state_dict(),
        "optimizer": optim_state,
        "retr_optimizer": retr_optim_state,
        "scheduler": scheduler.state_dict(),
        "retr_scheduler": retr_scheduler.state_dict() if retr_scheduler else None,
        "opt": opt,
        "retr_graph_optimizer": retr_graph_optim_state,
        "retr_graph_scheduler": retr_graph_scheduler.state_dict() if retr_graph_scheduler else None,
    }
    torch.save(checkpoint, fp)
    symlink_force(epoch_path, cp)
    if opt.save_optimizer and opt.shard_optim:
        optimizer._all_states = []

## edit by sai
## define a one layer MLP for projecting struct emb into text emb space
class ProjectorNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ProjectorNetwork, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out
import pandas as pd
import numpy as np
import csv
import json
import os
from model import custom_train
from model import custom_evaluate_adapretr_dynamic
from model import metrics_eval 

class InstructionFinetuning:
    def __init__(self):
        # code goes here
        print("InstructionFinetuning is called")

    def run_instructions_finetune(self, 
        train_retriever = True, 
        query_side_retriever_training=True, 
        reader_model_type= "google/t5-base-lm-adapt",
        model_path= "../step-4800/",
        train_data = ["../instructions_sample_train.jsonl"],
        eval_data = ["../instructions_sample.jsonl"], 
        name = "atlas-mlm-gen-S2ROC-220M-fos-instun",
        checkpoint_dir = "../experiments",
        train_steps = 10000,
        load_index_path = "../saved_index",
        load_subset_textindex = True,
        subset_textindex_name = "Physics,Bio-1,Art,History,Political-Science,Business,Economics,Geology,Computer-Science,Environmental-Science,Engineering,Med-1",
        task = "base",
        gold_score_mode = "pdist",
        per_gpu_batch_size =2 , 
        USE_CUSTOM_ARGS=False):

        if not USE_CUSTOM_ARGS:
            ## change default args over here
            train_retriever = True
            query_side_retriever_training=True
            reader_model_type= "google/t5-base-lm-adapt"
            model_path= "../step-4800/"
            train_data = ["../instructions_sample_train.jsonl"]
            eval_data = ["../instructions_sample.jsonl"]
            name = "atlas-mlm-gen-S2ROC-220M-fos-instun"
            checkpoint_dir = "../experiments"
            train_steps = 10000
            load_index_path = "../saved_index"
            load_subset_textindex = True
            subset_textindex_name = "Physics,Bio-1,Art,History,Political-Science,Business,Economics,Geology,Computer-Science,Environmental-Science,Engineering,Med-1"
            task = "base"
            gold_score_mode = "pdist"
            per_gpu_batch_size = 2

            print("DEFAULT ARGS ARE BEING USED")

            print("train_retriever:",train_retriever,"\n",
                "query_side_retriever_training",query_side_retriever_training,"\n",
                "reader_model_type",reader_model_type,"\n",
                "model_path",model_path,"\n",
                "train_data",train_data,"\n",
                "eval_data",eval_data, "\n",
                "name",name, "\n",
                "checkpoint_dir",checkpoint_dir,"\n",
                "train_steps",train_steps,"\n",
                "load_index_path",load_index_path,"\n",
                "load_subset_textindex",load_subset_textindex,"\n",
                "subset_textindex_name",subset_textindex_name, "\n",
                "task",task,"\n",
                "gold_score_mode",gold_score_mode,"\n",
                "per_gpu_batch_size",per_gpu_batch_size)
            
            custom_train.main(train_retriever,
                            query_side_retriever_training,
                            reader_model_type,
                            model_path,
                            train_data,
                            eval_data, 
                            name,
                            checkpoint_dir,
                            train_steps,
                            load_index_path,
                            load_subset_textindex,
                            subset_textindex_name,
                            task,
                            gold_score_mode,
                            per_gpu_batch_size) 
            
        else:
            print("CUSTOM ARGS ARE BEING USED")
            
            print("train_retriever:",train_retriever,"\n",
                "query_side_retriever_training",query_side_retriever_training,"\n",
                "reader_model_type",reader_model_type,"\n",
                "model_path",model_path,"\n",
                "train_data",train_data,"\n",
                "eval_data",eval_data, "\n",
                "name",name, "\n",
                "checkpoint_dir",checkpoint_dir,"\n",
                "train_steps",train_steps,"\n",
                "load_index_path",load_index_path,"\n",
                "load_subset_textindex",load_subset_textindex,"\n",
                "subset_textindex_name",subset_textindex_name, "\n",
                "task",task,"\n",
                "gold_score_mode",gold_score_mode,"\n",
                "per_gpu_batch_size",per_gpu_batch_size)
            
            custom_train.main(train_retriever,
                query_side_retriever_training,
                reader_model_type,
                model_path,
                train_data,
                eval_data, 
                name,
                checkpoint_dir,
                train_steps,
                load_index_path,
                load_subset_textindex,
                subset_textindex_name,
                task,
                gold_score_mode,
                per_gpu_batch_size) 

class Evaluation:
    def __init__(self):
        # code goes here
        print("Evaluation is called")
        
    def run_generation(self,
                reader_model_type= "google/t5-base-lm-adapt",
                model_path= "../step-9690/",
                eval_data = ["../instructions_sample.jsonl"], 
                name = "atlas-mlm-gen-S2ROC-220M-fosinstun-foseval-step9690-adapretrv2",
                checkpoint_dir = "../experiments",
                load_index_path = "../saved_index",
                load_subset_textindex = True,
                task = "base",
                gold_score_mode = "pdist",
                per_gpu_batch_size = 2,
                per_gpu_batch_size_domainindex= 600,
                no_sel_indices= 3,
                index_model_path = "../base/", 
                USE_CUSTOM_ARGS=False):
            
        if USE_CUSTOM_ARGS:
            print("CUSTOM ARGS ARE BEING USED")

            print(
                "reader_model_type",reader_model_type,"\n",
                "model_path",model_path,"\n",
                "eval_data",eval_data, "\n",
                "name",name, "\n",
                "checkpoint_dir",checkpoint_dir,"\n",
                "load_index_path",load_index_path,"\n",
                "load_subset_textindex",load_subset_textindex,"\n",
                "task",task,"\n",
                "gold_score_mode",gold_score_mode,"\n",
                "per_gpu_batch_size",per_gpu_batch_size,
                "per_gpu_batch_size_domainindex",per_gpu_batch_size_domainindex,
                "no_sel_indices",no_sel_indices,
                "index_model_path",index_model_path)
            
            custom_evaluate_adapretr_dynamic.main(
                reader_model_type,
                model_path,
                eval_data, 
                name,
                checkpoint_dir,
                load_index_path,
                load_subset_textindex,
                task,
                gold_score_mode,
                per_gpu_batch_size,
                per_gpu_batch_size_domainindex,
                no_sel_indices,
                index_model_path,
                ) 
                
        else:
            reader_model_type= "google/t5-base-lm-adapt",
            model_path= "../step-9690/",
            eval_data = ["../instructions_sample.jsonl"], 
            name = "atlas-mlm-gen-S2ROC-220M-fosinstun-foseval-step9690-adapretrv2",
            checkpoint_dir = "../experiments",
            load_index_path = "../saved_index",
            load_subset_textindex = True,
            task = "base",
            gold_score_mode = "pdist",
            per_gpu_batch_size = 2,
            per_gpu_batch_size_domainindex= 600,
            no_sel_indices= 3,
            index_model_path = "../base/"
            print("DEFAULT ARGS ARE BEING USED")

            print(
                "reader_model_type",reader_model_type,"\n",
                "model_path",model_path,"\n",
                "eval_data",eval_data, "\n",
                "name",name, "\n",
                "checkpoint_dir",checkpoint_dir,"\n",
                "load_index_path",load_index_path,"\n",
                "load_subset_textindex",load_subset_textindex,"\n",
                "task",task,"\n",
                "gold_score_mode",gold_score_mode,"\n",
                "per_gpu_batch_size",per_gpu_batch_size,
                "per_gpu_batch_size_domainindex",per_gpu_batch_size_domainindex,
                "no_sel_indices",no_sel_indices,
                "index_model_path",index_model_path)
            
            custom_evaluate_adapretr_dynamic.main(
                reader_model_type,
                model_path,
                eval_data, 
                name,
                checkpoint_dir,
                load_index_path,
                load_subset_textindex,
                task,
                gold_score_mode,
                per_gpu_batch_size,
                per_gpu_batch_size_domainindex,
                no_sel_indices,
                index_model_path,
                )     

    def run_metrics(self,
                    filepath,
                    outpath,
                    metric_name,
                    task_type,
                    paper_topic_file, 
                    USE_CUSTOM_ARGS=False):
        
        if USE_CUSTOM_ARGS:
            print("CUSTOM ARGS ARE BEING USED")

            print("filepath",filepath,"\n",
                  "outpath", outpath,"\n",
                  "metric_name", metric_name,"\n",
                   "task_type", task_type,"\n",
                   "paper_topic_file", paper_topic_file)

            metrics_eval.main(filepath,
                        outpath,
                        metric_name,
                        task_type,
                        paper_topic_file)
        else:
            print("DEFAULT ARGS ARE BEING USED")

            filepath="../instructions_sample_unified_v3_10.jsonl",
            outpath="../demo_inputs/",
            paper_topic_file="/home/evaluation/paper_topic_map.json",

            print("filepath",filepath,"\n",
                  "outpath", outpath,"\n",
                  "metric_name", metric_name,"\n",
                   "task_type", task_type,"\n",
                   "paper_topic_file", paper_topic_file)

            metrics_eval.main(filepath,
                        outpath,
                        metric_name,
                        task_type,
                        paper_topic_file)


        


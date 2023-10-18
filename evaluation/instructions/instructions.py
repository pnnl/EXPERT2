import pandas as pd
import numpy as np
import csv
import json
import os
from instructions.prompt import PromptGenerator


class InstructionsGenerator:
    def __init__(self, indir, outdir, task, config):
        self.indir = indir
        self.outdir = outdir
        self.task = task
        self.config = config
        self.col = self.config[self.task]['column']
        self.is_indir_list = False
        key_list  = list(self.config[self.task].keys())
        if 'categories' in key_list: self.categories = True
        else: self.categories = False     
        
        if 'label_type' in key_list: self.has_label_type = True
        else: self.has_label_type = False
        
        if self.col == '': self.json_list = pd.read_json(self.indir, lines=True)
        elif type(self.indir) == list:
            self.df = pd.read_json(self.indir[0], lines = True)
            self.df_doc_label = pd.read_csv(self.indir[1])
            self.is_indir_list = True
        else:   
            with open(self.indir, 'r') as json_file:
                self.json_list = list(json_file)
            
    def get_all_categories(self):
        
        if self.col == '': return None
        
        if self.is_indir_list:
            # create dict labels - 
            label_mapping = self.config[self.task]['label_mapping']
            label_mapping_dict= dict(zip([i for i in range(len(label_mapping))],label_mapping ))
            # Map labels in the dataframe
            self.df_doc_label[self.col] = self.df_doc_label['label'].map(label_mapping_dict)
            self.df_doc_label.rename(columns={self.config[self.task]['paper_col_name_1']:self.config[self.task]['paper_col_name_2']}, inplace=True)
            self.df_doc_label = self.df.merge(self.df_doc_label, on = [self.config[self.task]['paper_col_name_2']], how ='inner')
            self.json_list = self.df_doc_label.to_dict('records')
            return list(self.df_doc_label[self.col].unique())
        
        if self.categories: return list(self.config[self.task]['categories'].values())
        # get all the list of possible categories:
        all_cats = []

        # Iterate through all the json lines
        for json_str in self.json_list:
            result = json.loads(json_str) 
            if type(result[self.col]) == list : all_cats.extend(result[self.col])
            else: all_cats.append(result[self.col])
           
                
        # Flatten the list
        all_cats = [c for c in all_cats]

        # Get the unique list
        all_cats = list(set(all_cats))
        
        return all_cats
    
    # Function to create the instruction from template
    def get_prompt(self, TASK_CATS, result):
        # Define the instruction template
        return PromptGenerator.get_prompt(self.task, self.config[self.task]['task_type'], TASK_CATS, result)
    
    
    def generate_instructions(self,TASK_CATS):
        if TASK_CATS == None: return self.json_list
        # Initiate an empty list to save all the instructions
        if self.has_label_type: 
            instruction_dict = dict(zip(self.config[self.task]['label_type'].values(), [[] for i in range( len(self.config[self.task]['label_type']))]))
            # Iterate through the entire jsonl and generate instructions
            for json_str in self.json_list:
                result = json.loads(json_str)
                # Create the instruction
                train_json = {'query': self.get_prompt(TASK_CATS, result), 'target': result[self.col]}
                instruction_dict[self.config[self.task]['label_type'][result[self.config[self.task]['label_col']]]].append(train_json)
            return instruction_dict
        
        else: 
            instruction_list = []
            # Iterate through the entire jsonl and generate instructions
            for json_str in self.json_list:
                if type(json_str) == dict: result = json_str
                else: result = json.loads(json_str) 
                # Create the instruction
                res = result[self.col]
                if self.categories: 
                    res = self.config[self.task]['categories'][str(result[self.col])]
                train_json = {'query': self.get_prompt(TASK_CATS, result), 'target': res}
                instruction_list.append(train_json)
        
        return instruction_list

     
    def save_to_file(self, content):
        if self.has_label_type:
            for i, val in enumerate(list(self.config[self.task]['label_type'].values())):
                with open(self.outdir[i], "w") as f:
                    for instruction in content[val]:
                        f.write(json.dumps(instruction) + "\n")
        elif self.col == '': content.to_json(self.outdir, lines=True, orient = 'records')
        else:
            with open(self.outdir, "w") as f:
                for instruction in content:
                    f.write(json.dumps(instruction) + "\n")
        print("File(s) saved successfully")
        

        

            
                
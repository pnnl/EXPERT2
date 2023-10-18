'''

Created By: Shivam Sharma
Date: 12th Feb, 2023

This scripts contains classes and functions used for one-line generation of JSON result files for UQ_Pipeline.

'''

import os
import json
import torch

from datetime import datetime

from . import UQ_Entropy
from . import UQ_Generation

class UQ_Pipeline:

    '''
    
    This class contains the functions used to create JSON result files.

    Inputs:

        prompt (dtype: str): Prompt to generation text for
        model_name (dtype: str): Name or Path of model used for generation. Should be a huggingface style model
        tokenizer_path (dtype: str): Path to tokenizer. Only provide if custom model and tokenizer
        outpath (dtype: str): Path to save the JSON result files to
        gen_method (dtype: str): Generation Algorithm to use for generating. Please select from ['sampling', 'beam', 'nucleus']

        kwargs: Class also accepts other arguments to update which are used in UQ_Generation and UQ_Entropy classes

    '''
    
    def __init__(self, prompt, model_name, tokenizer_path=None, outpath = "./outputs/", gen_method="sampling", **kwargs):
        
        self.prompt = prompt
        self.model_name = model_name
        self.tokenizer_path = tokenizer_path
        self.gen_method = gen_method
        self.args = kwargs
        self.outpath = outpath
        
        # Make output directory if not present
        os.makedirs(self.outpath, exist_ok=True)
        
        # Initialize UQ_Generation class
        self.gen_pipeline = UQ_Generation(self.model_name, self.tokenizer_path)
    
    def get_generation(self):

        '''
        
        This function generates tokens for the provided input prompt.

        Output:
            generated_output (dtype: dict): Contains Sequences and Scores produced when generating output tokens for the given input
        
        '''
        
        # Sampling Method
        if self.gen_method == "sampling":
            self.generated_outputs = self.gen_pipeline.gen_sampling(self.prompt, **self.args)
        
        # Beam Search with NGram Method
        elif self.gen_method == 'beam':
            self.generated_outputs = self.gen_pipeline.gen_beam(self.prompt, **self.args)
        
        # Nucleus (Top-P) Sampling Method
        elif self.gen_method == "nucleus":
            self.generated_outputs = self.gen_pipeline.gen_nucleus(self.prompt, **self.args)
        
        else:
            raise ValueError(f"{self.gen_method} not available. Please select from ['sampling', 'beam', 'nucleus']")
        
        return self.generated_outputs
    
    def get_entropy(self):

        '''
        
        This function generates the 4 different entropy scores for the generated sequences.

        Output:
            entropy_set (dtype: str): Contain the final four entropy scores in the order 
                        (predictive entropy, normalized predictive entropy, lexical similarity, semantic entropy)
        
        '''
        
        # Initialize UQ_Entropy Class
        self.gen_probs = self.gen_pipeline.get_probab(self.generated_outputs)
        self.entropy_pipeline = UQ_Entropy(self.gen_probs)
        
        # 1. Entropy
        self.entropy = self.entropy_pipeline.get_entropy()
        print("\tBasic Entropy Done.....")

        # 2. Normalized Entropy
        self.norm_entropy = self.entropy_pipeline.normalized_entropy()
        print("\tNormalized Entropy Done.....")
        
        # 3. Lexical Similarity
        self.lex_sim = self.entropy_pipeline.lexical_similarity(self.gen_pipeline.gen_sequences, 
                                                                self.gen_pipeline.tokenizer)
        print("\tLexical Similarity Done.....")
        
        # 4. Semantic Uncertertainty
        self.sem_uncertainty = self.entropy_pipeline.semantic_uncertainty(self.prompt, 
                                                                          self.gen_pipeline.gen_sequences, 
                                                                          self.gen_pipeline.tokenizer)
        print("\tSemantic Uncertainity Done.....")
        
        return (self.entropy, self.norm_entropy, self.lex_sim, self.sem_uncertainty)
    
    def save_json(self, save_json=True):
        
        '''
        
        This function generates and saves the final results JSONS to the provided output path.

        Input:
            save_json (dtype: bool): Whether to save the final result JSONs to outpath or not.

        Output:
            entropy_set (dtype: dict): Contains the final results for the given prompt and generation style.
        
        ''' 

        out_json = {}
        
        # Start Generation
        _ = self.get_generation()
        print("Generation Done.....")
        
        # Start Entropy Calculation
        _ = self.get_entropy()
        print("Entropy Done.....")

        out_json["Generation"] = {}
        
        # Save Different Entropy Scores
        out_json["Entropy"] = str(self.entropy)
        out_json["Normalized_Entropy"] = str(self.norm_entropy)
        out_json["Lexical_Similarity"] = str(self.lex_sim)
        out_json["Semantic Uncertainty"] = str(self.sem_uncertainty)
       
        # Save Generation Algo and Model Name/Path
        out_json["Gen_Algo"] = self.gen_method
        out_json["Model"] = self.model_name
       
        # Save Input Prompt/Question
        out_json["Input_Question"] = self.prompt

        self.seq_probs = torch.stack(self.generated_outputs.scores, dim=1).softmax(-1)
        
        # Save Generations and corresponding probablity sets
        for seq_num, seq in enumerate(self.gen_pipeline.gen_sequences):
            
            gen_text = self.gen_pipeline.tokenizer.decode(seq, skip_special_tokens=True).replace("\n","")

            if len(gen_text) == 0:
                continue
            
            out_json["Generation"][gen_text] = []
                
            for i, token_probs in enumerate(self.seq_probs[seq_num]):

                gen_token = self.gen_pipeline.gen_sequences[seq_num][i]
                gen_word = self.gen_pipeline.tokenizer.decode(gen_token)
                
                if '\n' in gen_word:
                    continue
                
                token_probs = token_probs.sort(descending=True)
                
                sample_list, prob_list = [], []
                
                for new_idx in range(10):
                    
                    curr_token = token_probs[1][new_idx]
                    curr_word = self.gen_pipeline.tokenizer.decode(curr_token)
                    
                    curr_prob = token_probs[0][new_idx]
                    
                    sample_list.append(curr_word)
                    prob_list.append(str(float(curr_prob)))
                
                out_json["Generation"][gen_text].append((gen_word, sample_list, prob_list))

        # The Final JSON File is Saved as Current Date and Time
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S").replace("/","_").replace(":", "_").replace(" ", "_")
        
        out_file = f"{self.outpath}/{dt_string}.json"
        print(f"Output JSON saved at:\n{out_file}")
        
        if save_json:
            with open(out_file, 'w') as fp:
                json.dump(out_json, fp)
        
        return out_json
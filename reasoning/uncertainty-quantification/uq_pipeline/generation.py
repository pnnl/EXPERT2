'''

Created By: Shivam Sharma
Date: 10th Feb, 2023

This scripts contains classes and functions used for different generation steps in the Uncertainty Quantification Pipeline

'''

import os
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GPT2LMHeadModel

class UQ_Generation:

    '''
    
    This class contains the various functions used to generate text for a given input and provide the probablity distribution for 
    each generation.

    Inputs:
        model_name (dtype: str): Name or Path of model used for generation. Should be a huggingface style model
        tokenizer_name (dtype: str): Path to tokenizer. Only provide if custom model and tokenizer
    
    Citation:
        To read more about Generation Algorthms used in this script, please refer to the following link
        https://huggingface.co/blog/how-to-generate
    
    '''
    
    def __init__(self, model_name, tokenizer_name=None):
        
        # Initialize Model
        if 'gpt' in model_name:
            self.model = GPT2LMHeadModel.from_pretrained(model_name, return_dict_in_generate=True)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, return_dict_in_generate=True)
            
        # Initialize Tokenizer
        if tokenizer_name != None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def  gen_sampling(self, prompt, **kwargs):
        
        '''
        
        This function generates tokens for a given prompt using Sampling Generation Algorithm.

        Inputs:
            prompt (dtype: str): Prompt to generation text for
            
            kwargs:
                temp (dtype: float): Temperature value for generation
                num_seq (dtype: int): Total number of sequences to be generated
                num_tokens (dtype: int): Maximum number of tokens to be generated in each sequence
        
        Output:
            generated_output (dtype: dict): Contains Sequences and Scores produced when generating output tokens for the given input
        
        '''

        # Set Temperature value from kwargs
        if 'temp' in kwargs:
            temp = kwargs['temp']
        else:
            temp = 0.5
        
        # Set Number of Sequences from kwargs
        if 'num_seq' in kwargs:
            num_seq = kwargs['num_seq']
        else:
            num_seq = 10
                
        # Set Number of Tokens from kwargs
        if 'num_tokens' in kwargs:
            num_tokens = kwargs['num_tokens']
        else:
            num_tokens = 15
        
        # Convert Prompt into Input IDs
        self.input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        
        # Generate Tokens for Input IDs
        generated_outputs = self.model.generate(self.input_ids, do_sample=True,
                                                temperature=temp,num_return_sequences=num_seq, top_k=0,
                                                output_scores=True, max_length=len(self.input_ids[0]) + num_tokens)
        
        return generated_outputs

    def gen_beam(self, prompt, **kwargs):
        
        '''
        
        This function generates tokens for a given prompt using Beam Search Generation Algorithm.

        Inputs:
            prompt (dtype: str): Prompt to generation text for
            
            kwargs:
                num_beams (dtype: int): Number of beams for the generation
                temp (dtype: float): Temperature value for generation
                num_seq (dtype: int): Total number of sequences to be generated. For this generation algorithm, 
                                      should be less than or equal to num_beams value.
                num_tokens (dtype: int): Maximum number of tokens to be generated in each sequence
                num_ngrams (dtype: int): Maximum NGram length to prevent from repeating
            
        Output:
            generated_output (dtype: dict): Contains Sequences and Scores produced when generating output tokens for the given input
        
        '''

        # Set Number of Beams from kwargs
        if 'num_beams' in kwargs:
            num_beams = kwargs['num_beams']
        else:
            num_beams = 15
        
        # Set Temperature value from kwargs
        if 'temp' in kwargs:
            temp = kwargs['temp']
        else:
            temp = 0.5
        
        # Set Number of Sequences from kwargs
        if 'num_seq' in kwargs:
            num_seq = kwargs['num_seq']
        else:
            num_seq = 10
                
        # Set Number of Tokens from kwargs
        if 'num_tokens' in kwargs:
            num_tokens = kwargs['num_tokens']
        else:
            num_tokens = 15
        
        # Set Number of N-Grams from kwargs
        if 'num_ngram' in kwargs:
            num_tokens = kwargs['num_ngram']
        else:
            num_ngram = 2
        
        # Convert Prompt into Input IDs
        self.input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        
        # Generate Tokens for Input IDs
        generated_outputs = self.model.generate(self.input_ids, max_length=len(self.input_ids[0]) + num_tokens, 
                                                output_scores=True, temperature=temp, 
                                                no_repeat_ngram_size=num_ngram,
                                                num_return_sequences=num_seq, num_beams=num_beams)

        return generated_outputs
    
    def gen_nucleus(self, prompt, **kwargs):
        
        '''
        
        This function generates tokens for a given prompt using Nucleus (Top-P) Sampling Generation Algorithm.

        Inputs:
            prompt (dtype: str): Prompt to generation text for
            
            kwargs:
                num_seq (dtype: int): Total number of sequences to be generated
                num_tokens (dtype: int): Maximum number of tokens to be generated in each sequence
                top_p (dtype: float): Top-P value for generation. Should be less than 1
        
        Output:
            generated_output (dtype: dict): Contains Sequences and Scores produced when generating output tokens for the given input
        
        '''

        # Set Number of Sequences from kwargs
        if 'num_seq' in kwargs:
            num_seq = kwargs['num_seq']
        else:
            num_seq = 10
                
        # Set Number of Tokens from kwargs
        if 'num_tokens' in kwargs:
            num_tokens = kwargs['num_tokens']
        else:
            num_tokens = 15
        
        # Set Top-P value from kwargs
        if 'top_p' in kwargs:
            top_p = kwargs['num_ngram']
        else:
            top_p = 0.9
        
        # Convert Prompt into Input IDs
        self.input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        
        # Generate Tokens for Input IDs
        generated_outputs = self.model.generate(self.input_ids, output_scores=True,
                                                max_length=len(self.input_ids[0]) + num_tokens,
                                                do_sample=True, top_p=top_p, top_k=0, num_return_sequences=num_seq)
        
        return generated_outputs
    
    def get_probab(self, generated_outputs):

        '''
        
        This function calcualtes the probablity values for the generated set of tokens.

        Input:
            generated_output (dtype: dict): Contains Sequences and Scores produced when generating output tokens for the given input
        
        Output:
            gen_probs (dtype: list): Contains the probablity ditribution for each generated sequences
        
        '''
        # Seperate Generated Sequence from Model Output
        self.gen_sequences = generated_outputs.sequences[:, self.input_ids.shape[-1]:]

        # Calcualte Probablity Distribution from Model Output
        seq_probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)
        
        # Final Probablity List for Generated Sequence
        gen_probs = torch.gather(seq_probs, 2, self.gen_sequences[:, :, None]).squeeze(-1)
        
        return gen_probs

    def get_gen_text(self, gen_sequences = None, tokenizer = None):

        '''
        
        This function converts the generated tokens into text.

        Inputs:
            gen_sequences (dtype: list): List containing tokens values for each generated sequence
            tokenizer (dtype: HuggingFace Tokenizer): If gen_sequence provided, will use this tokenizer
        
        Output:
            gen_text_list (dtype: list): List of generated strings
        
        '''
        # Convert user-defined Generated Sequences into Text
        if gen_sequences != None:
            gen_text_list = []

            for seq in gen_sequences:
                gen_text = tokenizer.decode(seq, skip_special_tokens=True)
                if len(gen_text) != 0:
                    gen_text_list.append(gen_text)
        
        # Convert In-Class Generated Sequnces into Text
        else:
            gen_text_list = []
            
            for seq in self.gen_sequences:
                gen_text = self.tokenizer.decode(seq, skip_special_tokens=True)
                if len(gen_text) != 0:
                    gen_text_list.append(gen_text)
        
        return gen_text_list
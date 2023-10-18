'''

Created By: Shivam Sharma
Date: 10th Feb, 2023

This scripts contains classes and functions used for different entropy calculation steps 
in the Uncertainty Quantification Pipeline

'''

import torch
import evaluate
import numpy as np

from scipy.stats import entropy
from transformers import AutoTokenizer, DebertaForSequenceClassification, GPT2LMHeadModel

class UQ_Entropy:
    
    '''
    
    This class contains the various functions used for calculating the various entropy scores for a given set of generations.

    Input:
        gen_probs (dtype: list): List of probablity distribution for generated sequences
    
    '''

    def __init__(self, gen_probs):
        
        # Generation Probablities
        self.gen_probs = gen_probs
        
    def get_entialment(self, seq1, seq2):

        '''
        
        This function uses an Natural Language Inference (NLI) model to predict whether two provided sequences are in 
        "contradiction", are "neutral", or are in "entailment". This function is used for calculating Semantic Entropy Score.

        Inputs:
            seq1 (dtype: str): First Sequence to be evaluated
            seq2 (dtype: str): Second Sequence to be evaluated
        
        Output:
            candidate_ans (dtype: str): Entailment between the two strings. Will be one ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]
        
        '''
        
        # List of Candidate Answers for the Entailment Model
        candidate_ans = ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]
        
        prompt = f"{seq1}[SEP]{seq2}"
        
        # Passing the Set of Sequences through NLI Model
        inputs = self.nli_tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
        
        return candidate_ans[logits.argmax().item()]
    
    def get_entropy(self, min_val=0.7, max_val=3):

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
        ent_val = np.mean([entropy(prob_list) for prob_list in self.gen_probs])

        # Range-Normalization
        ent_val = (ent_val - min_val)/(max_val - min_val)
        
        # Checks to keep score between 0 and 1
        if ent_val < 0:
            ent_val = ent_val * (-1)

        if ent_val > 1:
            ent_val = 1

        return ent_val

    def normalized_entropy(self, min_val=0.05, max_val=0.15):
        
        '''
        
        This function returns normalized predictive entropy score, normalized over a user-defined or pre-defined range.
        We use the Shanon Entropy to calcualte the predictive entropy for a given set of generated sequences.

        Input:
            min_val (dtype: float): The minimum value for normalization over range
            max_val (dtype: float): The maximum value for normalization over range
        
        Output:
            norm_entropy: Range-Normlized and Token-Normlized Predictive Entropy score for generated sequences
        
        Citation:
            Andrey Malinin and Mark Gales. Uncertainty estimation in autoregressive structured prediction.
            arXiv preprint arXiv:2002.07650, 2020
        
        '''
        # Mean Token-Normalized Predictive Entropy
        norm_entropy = np.mean([entropy(prob_list)/len(prob_list) for prob_list in self.gen_probs])

        # Range-Normalization
        norm_entropy = (norm_entropy - min_val)/(max_val - min_val)

        # Checks to keep score between 0 and 1
        if norm_entropy < 0:
            norm_entropy = norm_entropy * (-1)

        if norm_entropy > 1:
            norm_entropy = 1
        
        return norm_entropy
    
    def lexical_similarity(self, gen_sequences, tokenizer, min_val=1, max_val=13):
        
        '''
        
        This function returns entropy score calculated using lexical similarity, normalized over a user-defined or pre-defined range.
        We use the Rouge-L score to calcualte the similarity between a set of generated sequences. We inverse the similarity score
        to represent the entropy.

        Input:
            gen_sequences (dtype: list): List of generated sequences
            tokenizer (dtype: Huggingface Tokenizer): HuggingFace tokenizer for the model used to generate the given sequences
            min_val (dtype: float): The minimum value for normalization over range
            max_val (dtype: float): The maximum value for normalization over range
        
        Output:
            lexical_sim: Range-Normlized and Inversed Lexical Similarity score for generated sequences
        
        Citation:
            Marina Fomicheva, Shuo Sun, Lisa Yankovskaya, Fr´ed´eric Blain, Francisco Guzm´an, Mark Fishel,
            Nikolaos Aletras, Vishrav Chaudhary, and Lucia Specia. Unsupervised quality estimation for
            neural machine translation. Transactions of the Association for Computational Linguistics, 8:
            539–555, 2020. 2, 8
        
        '''
        
        # Initializing Rouge Calculator
        rouge = evaluate.load("rouge")

        # Converting Tokens into Text
        self.gen_text = [tokenizer.decode(seq, skip_special_tokens=True) for seq in gen_sequences]
        
        # Calculating Similarity using Rouge-L
        rouge_sum = 0
        for i, s1 in enumerate(self.gen_text[:-1]):
            for s2 in self.gen_text[i+1:]:
                score = rouge.compute(predictions=[s1],references=[s2])['rougeL']
                rouge_sum += score

        C = len(self.gen_text) * (len(self.gen_text) - 1)/2

        # Final Lexical-Similarity Score
        lexical_sim = rouge_sum/C

        # Inversing to Estimate Entropy
        lexical_sim = 1/lexical_sim

        # Range-Normalization
        lexical_sim = (lexical_sim - min_val)/(max_val - min_val)

        # Checks to keep score between 0 and 1
        if lexical_sim < 0:
            lexical_sim = lexical_sim * (-1)

        if lexical_sim > 1:
            lexical_sim = 1

        return lexical_sim
    
    def semantic_uncertainty(self, ques, gen_sequences, tokenizer, print_ent=False, min_val=-85, max_val=-30):
        
        '''
        
        This function return the Semantic Entropy Score, normalized over a user-defined or pre-defined range.
        We use the DeBERTa-Large model, fine-tuned on the MNLI dataset, to estimate the entailment between the generations.

        Input:
            ques (dtype: str): Input prompt/question to the generation model
            gen_sequences (dtype: list): List of generated sequences
            tokenizer (dtype: Huggingface Tokenizer): HuggingFace tokenizer for the model used to generate the given sequences
            print_ent (dtype: bool): Verbose model to print the entailment sequence and results
            min_val (dtype: float): The minimum value for normalization over range
            max_val (dtype: float): The maximum value for normalization over range
        
        Output:
            sem_ent: Range-Normlized Semantic Entropy score for generated sequences
        
        Citation:
            Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic Uncertainty: Linguistic Invariances for 
            Uncertainty Estimation in Natural Language Generation. arXiv preprint arXiv:2302.09664.
        
        '''

        nli_model = "microsoft/deberta-large-mnli"
        
        # Initializing NLI Model
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model)
        self.nli_model = DebertaForSequenceClassification.from_pretrained(nli_model)
        
        # Calculating Sequence Probablity
        prob_list = [seq_probs.numpy().prod() for seq_probs in self.gen_probs]

        # Converting Tokens into Text
        self.gen_text = [tokenizer.decode(seq, skip_special_tokens=True) for seq in gen_sequences]

        # Creating Meaning Set
        self.meaning_set = [[self.gen_text[0]]]

        # Populating Meaning Set
        for s1 in self.gen_text[1:]:  
            for group_idx, group in enumerate(self.meaning_set):
                s2 = group[0]
                
                seq1 = f"{ques}|{s1}"
                seq2 = f"{ques}|{s2}"
                
                ent_left = self.get_entialment(seq2,seq1)
                ent_right = self.get_entialment(seq1, seq2)
                
                if print_ent:
                    print("\n ############# \n")
                    print(f"Prompt: {seq2}[SEP]{seq1}\n")
                    print(ent_left)
                    print(f"\nPrompt: {seq1}[SEP]{seq2}\n")
                    print(ent_right)
                
                # If Bi-Directional Entailment -> Add to Existing set
                if ent_left == "ENTAILMENT" and ent_right == "ENTAILMENT":
                    self.meaning_set[group_idx].append(s1)
                    sem_sim = True
                    break
                
                sem_sim = False
            # If No Bi-Directional Entailment -> Create New Set
            if not sem_sim:
                self.meaning_set.append([s1])
        
        self.sem_probs = []

        # Probablity List for Meaning Set
        for seq_set in self.meaning_set:
            idx_set = [self.gen_text.index(seq) for seq in seq_set]
            prob_set = [prob_list[idx] for idx in idx_set]
            
            self.sem_probs.append(sum(prob_set))

        #  Mean Entropy for Meaning Set
        self.log_prob = [np.log(group_prob) for group_prob in self.sem_probs if group_prob != 0]
        sem_ent = np.mean(self.log_prob)

        # Range-Normalization
        sem_ent = (sem_ent - min_val)/(max_val - min_val)
        
        # Checks to keep score between 0 and 1
        if sem_ent < 0:
            sem_ent = sem_ent * (-1)
        
        if sem_ent > 1:
            sem_ent = 1

        return sem_ent
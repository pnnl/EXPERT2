import pandas as pd
# from sentence_transformers import SentenceTransformer
import glob, os
import json
import numpy as np
from itertools import groupby
import sys
from sklearn import metrics
import logging
logger = logging.getLogger(__name__)

def get_em_accuracy(gold_labels, gen_labels):
    correct = 0
    incorrect = 0
    for gold, gen in zip(gold_labels, gen_labels):
        if gold==gen:
            correct+=1
        else:
            incorrect+=1
    return correct / (correct + incorrect) * 100


def get_alo_accuracy(gold_labels, gen_labels):
    correct = 0
    incorrect = 0
    for gold, gen in zip(gold_labels, gen_labels):
        gold = gold.split(",")
        gen = gen.split(",")
        if len(set(gold).intersection(set(gen))) >0:
            correct+=1
        else:
            incorrect+=1
    return correct / (correct + incorrect) * 100


def get_pm_accuracy(gold_labels, gen_labels):
    correct = 0
    incorrect = 0
    for gold, gen in zip(gold_labels, gen_labels):
        gold = gold.split(",")
        gen = gen.split(",")
        intrsct_len = len(set(gold).intersection(set(gen)))
        if intrsct_len >0:
            correct+= (intrsct_len / len(set(gold)))
        else:
            incorrect+=1
    return correct / (correct + incorrect) * 100

def get_f1_score(gold_labels, gen_labels):
    y_true = gold_labels
    y_pred = gen_labels
    overall_metrics = metrics.classification_report(y_true, y_pred, digits=2,output_dict=True)
    return overall_metrics['weighted avg']['f1-score']

def get_rel_score(df, outfile):
    # Cosine Similarity
    def cosine_similarity(vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        similarity = dot_product / (norm_vector1 * norm_vector2)
        return similarity
    
    
    def get_passage_level_similarity(answer, passages):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        cos_sim = []
        for passage in passages:     
            sentences = [answer, passage['title']]
            embeddings = model.encode(sentences)
            cos_sim.append(cosine_similarity(embeddings[0], embeddings[1]))
        return cos_sim

#         df = pd.read_json("/shared/rcfs/expert/data/text_data/test_atlas/experiments/atlas-mlm-gen-Atlas-220M-fos-instun/instructions_sample-step-1000_selectedindex.jsonl", lines = True)
        # Add two columns - passage_level_similarity and agg_similarity
    df['passage_level_similarity'] = ''
    df['agg_similarity'] = ''
    for index, row in df.iterrows():
        # print(index)
        logger.info(f"data iterator done: {index}")
        answers = row['answers'][0].split(',')
        passage_level_similarity = {}
        agg_similarity = {}
        if type(answers)==list:
            for ans in answers:
                passage_level_similarity[ans] = get_passage_level_similarity(ans, row['passages'])
                agg_similarity[ans] = sum(passage_level_similarity[ans]) / len(passage_level_similarity[ans])
        else:
            passage_level_similarity[answers] = get_passage_level_similarity(answers, row['passages'])
            agg_similarity[answers] = sum(passage_level_similarity[answers]) / len(passage_level_similarity[answers])
            

        df.at[index, 'passage_level_similarity'] = passage_level_similarity
        df.at[index, 'agg_similarity'] = agg_similarity
        pd.DataFrame(df.iloc[index]).T.to_json(os.path.join(outpath,f"{outfile}_relevance_metric.jsonl"), orient='records', lines = True, mode = 'a')
    # print("**File Saved**")


## get custom semantic relevance score
def get_rel_score_customtopic(df_input, outfile, topiclist=None):
    # Cosine Similarity
    def cosine_similarity(vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        similarity = dot_product / (norm_vector1 * norm_vector2)
        return similarity
    
    
    def get_passage_level_similarity(answer, passages):    
        cos_sim = []
        for passage in passages:     
            sentences = [answer, passage['title']]
            embeddings = model.encode(sentences)
            cos_sim.append(cosine_similarity(embeddings[0], embeddings[1]))
        return cos_sim
    
    def get_passage_level_similarity_dict(answer, passages_row):
        return {ans: get_passage_level_similarity(ans, passages_row ) for ans in answer}
    
    def get_agg_similarity(passage_level_similarity_row):
        return {ans: sum(passage_level_similarity_row[ans]) / len(passage_level_similarity_row[ans]) for ans in passage_level_similarity_row }

    df = df_input.copy()

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    if topiclist == None:
        df['passage_level_similarity'] = df.apply(lambda x : get_passage_level_similarity_dict(x['answers'][0].split(','), x['passages']), axis=1)
        df['agg_similarity'] = df['passage_level_similarity'].apply(lambda x : get_agg_similarity(x))
    else:
        df['passage_level_similarity'] = df['passages'].apply(lambda x : get_passage_level_similarity_dict(topiclist, x))
        df['agg_similarity'] = df['passage_level_similarity'].apply(lambda x : get_agg_similarity(x))
    
    df.to_json(os.path.join(outpath,f"{outfile}_sem_relevance_metric.jsonl"), orient='records', lines=True, mode='a')
    logger.info("saved df")


def get_rel_score_topic(df, outfile, task_type, paper_topic_file, topiclist=None):
    print("\n\n Generating rel score topic\n")
    # load topic-paper map file 
    f = open(paper_topic_file)
    paper_topic_dict = json.load(f)
    df['paper_id_list'] = df['passages'].apply(lambda x: [x_val['id'].split('S2ORC_')[0]+'S2ORC' for x_val in x])
    df['answers'] = df['answers'].apply(lambda x: x[0].split(','))

    def get_topic_list(paper_list, name):
        print(name)
        return [paper_topic_dict[paper_id] for paper_id in paper_list]

    def rel_score_topic(topic_list, answers):
        filtered_list = list(filter(lambda x: x in answers, topic_list))
        return len(filtered_list)/len(topic_list)
    print("\n\n Generating topic list\n")
    df['topic_list'] = df.apply(lambda x: get_topic_list(x.paper_id_list, x.name), axis=1)
    print("\n\n Generating rel score\n")

    if task_type=="fos":
        df['relevance_score_topic'] = df.apply(lambda x: rel_score_topic(x.topic_list,x.answers), axis = 1)
    elif task_type=="mesh-desc":
        df['relevance_score_topic'] = df.apply(lambda x: rel_score_topic(x.topic_list,topiclist), axis = 1)

    df.to_json(os.path.join(outpath,f"{outfile}_relevance_metric_topic.jsonl"), orient='records', lines = True)
    print("**File Saved**")

if __name__ == "__main__":
    # logger.info("main is called")
    # print("main is called")
    filepath = sys.argv[1]
    outpath = sys.argv[2]
    metric_name = sys.argv[3]
    task_type = sys.argv[4]
    paper_topic_file = sys.argv[5]
    
    df =  pd.read_json(filepath, lines=True)
    

    topiclist = ['Biology','Medicine']
    # topiclist = None # use answers from the data
    
    if metric_name=="semantic":
        logger.info("rel score is called")
        # print("rel score is called")
        # get_rel_score(df, os.path.basename(filepath).split('.')[0])
        get_rel_score_customtopic(df, os.path.basename(filepath).split('.')[0],topiclist)

    elif metric_name=="topic":    
        get_rel_score_topic(df, os.path.basename(filepath).split('.')[0], task_type, paper_topic_file, topiclist)

    elif metric_name=="accuracy":
        # Correct Answers
        gold_labels = df['answers'].tolist()
        gold_labels = [it[0] for it in gold_labels]

        # Answers generated by model
        gen_labels = df['generation'].tolist()

        metric_res = {}
        # gold_labels = ['testA', 'testB,testC', 'testD,testF,testE', 'testX,testY']
        # gen_labels =  ['testB', 'testC',       'testD,testF',       'testX,testY,testZ']
        metric_res['exact_match'] = [get_em_accuracy(gold_labels, gen_labels)]
        metric_res['atleast_one_match'] = [get_alo_accuracy(gold_labels, gen_labels)]
        metric_res['weighted_match'] = [get_pm_accuracy(gold_labels, gen_labels)]
        metric_res['f1_weighted_avg'] = [get_f1_score(gold_labels, gen_labels)]

        res_df = pd.DataFrame(metric_res)

        outfile = os.path.splitext(os.path.basename(filepath))[0]

        res_df.to_json(os.path.join(outpath,f"{outfile}_metrics.jsonl"), orient='records', lines = True)

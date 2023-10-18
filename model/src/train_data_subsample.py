import pandas as pd 
import sys 
import os

#Sample size for each topic  
SAMPLE_SIZE = 5000000
OUT_FILE = "S2ORC_gensci_train_5m.jsonl"

outdir = f"/rcfs/projects/expert/data/embeddings/NonPro/train/{OUT_FILE}"
indir = "/rcfs/projects/expert/data/S2ORC_gensci/S2ORC_gensci_MLKG/tmp/topicwise_text_segments/"

chunksize = 1000000
i = 0
max_count = SAMPLE_SIZE/chunksize

topic_file_list = os.listdir(indir)

for file in topic_file_list:
    i = 0
    if '.jsonl' in file:
        print(f"Processing file -- {file}\n")
        with pd.read_json(os.path.join(indir, file), lines = True, chunksize=chunksize) as reader:
            for df in reader:
                print(f"Processing chunk -- {i}\n")
                i+=1
                df.to_json(outdir, lines = True, orient = 'records', mode='a')
                if i == max_count: break

print('*****File saved*****')
        
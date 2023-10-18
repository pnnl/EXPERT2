import argparse
import os
import pickle
import random

filepath = "/rcfs/projects/expert/wagl687/output/sequences/test_run/opt-125m_generations.pkl"

with open(filepath, 'rb') as infile:
    sequences = pickle.load(infile)

print(len(sequences))
print(sequences[0])
print(sequences[0].keys)
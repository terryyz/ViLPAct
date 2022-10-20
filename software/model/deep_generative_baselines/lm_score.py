import argparse
import torch
import json
import joblib
import pickle
import numpy as np
from tqdm import tqdm
from transformers import ProphetNetTokenizer
from generator import Generator

with open("action_level_vocab.txt","r") as f:
    vocabs = f.read().splitlines()
vocab_dict = {vocab.split()[0]: vocab.split()[1] for vocab in vocabs}

def score(tokens_tensor):
    log_probs = model.get_normalized_probs(tokens_tensor, log_probs=True) # log_probs
    return log_probs.cpu().detach().numpy()

def encode(sentence):
    return torch.tensor([[int(vocab_dict[word]) for word in sentence.split()]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Charades Training')
    parser.add_argument('--in-dir', default='action_level_planning')
    args = parser.parse_args()
    results = dict()
    generator = Generator(f"{args.in_dir}/processed",f"{args.in_dir}/finetune/model/checkpoint_best.pt","prophetnet", 'hMoEup')
    model = generator.models[0]
    src = pickle.load(open(f"planning_test_src", "rb"))
    choices = pickle.load(open(f"choices", "rb"))
    t_test_src = pickle.load(open(f"t_test_src", "rb"))
    for k, choice in choices.items():
        tmp=[]
        for c in choice:
            tokens_tensor = encode(t_test_src[k] + " [SEP] " + src[k].split(" [SEP] ")[1]+" "+" ".join(c))
            scr=score(tokens_tensor)
            tmp.append({" ".join(c):[scr,sum(scr)]})
            torch.cuda.empty_cache()
        results[k] = tmp

    with open(f"charades/{args.in_dir}/test_log_probs.pkl", "wb") as f:
        pickle.dump(results,f)

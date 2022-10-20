import os
import pickle
import argparse
import nltk
import json
import numpy as np
from tqdm import tqdm
from numpyencoder import NumpyEncoder

def mean(l):
    return sum(l)/len(l)

def calculate_bleu_scores(references, hypotheses):
    """
    Calculates BLEU 1-4 scores based on NLTK functionality
    Args:
        references: List of reference sentences
        hypotheses: List of generated sentences
    Returns:
        bleu_1, bleu_2, bleu_3, bleu_4: BLEU scores
    """
    bleu_1 = np.round(100 * nltk.translate.bleu_score.corpus_bleu(references, hypotheses, weights=(1.0, 0., 0., 0.)), decimals=2)
    bleu_2 = np.round(100 * nltk.translate.bleu_score.corpus_bleu(references, hypotheses, weights=(0.50, 0.50, 0., 0.)), decimals=2)
    bleu_3 = np.round(100 * nltk.translate.bleu_score.corpus_bleu(references, hypotheses, weights=(0.34, 0.33, 0.33, 0.)), decimals=2)
    bleu_4 = np.round(100 * nltk.translate.bleu_score.corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)), decimals=2)
    return bleu_1, bleu_2, bleu_3, bleu_4

class Evaluator(object):
    def __init__(self, args):
        self.in_dir = args.in_dir
#         action_src_reference = pickle.load(open(f"{self.in_dir}/test_src_action.pkl","rb"))
        action_tgt_reference = pickle.load(open(f"{self.in_dir}/test_tgt_gt.pkl","rb"))
        action_tgt_candidate = pickle.load(open(f"{self.in_dir}/test_pred_10.pkl","rb"))
        action_src_candidate = pickle.load(open(f"{self.in_dir}/test_src_action.pkl","rb"))
        action_src_reference = pickle.load(open(f"{self.in_dir}/test_src_gt.pkl","rb"))
        self.action_class = pickle.load(open(args.action_class_file,"rb"))
        self.ref = [action_src + action_ids for action_src, action_ids in zip(action_src_reference, action_tgt_reference)]
        self.cand = [[src_actions + tgt_action[0].split(" ") for tgt_action in tgt_actions] for src_actions, tgt_actions in zip(action_src_candidate, action_tgt_candidate)]
    def evaluate(self):
        avg_recall,avg_precision,avg_item_seq_accu = [],[],[]
        
        reference,candidate = [],[]
        for j, (ref, cands) in tqdm(enumerate(zip(self.ref, self.cand))):
            tmp_recall,tmp_precision,tmp_item_seq_accu = [],[],[]
            for cand in cands:
                try:
                    tmp_recall.append(len(set(ref) & set(cand))/ len(set(ref)))
                except:
                    tmp_recall.append(0)
                try:
                    tmp_precision.append(len(set(ref) & set(cand))/ len(set(cand)))
                except:
                    tmp_precision.append(0)
                try:
                    tmp_item_seq_accu.append(sum([1 for i in range(min(len(ref),len(cand))) if cand[i] == ref[i]])/len(cand))
                except:
                    tmp_item_seq_accu.append(0)
                reference.append([ref])
                candidate.append(cand)
            avg_recall.append(mean(tmp_recall))
            avg_precision.append(mean(tmp_precision))
            avg_item_seq_accu.append(mean(tmp_item_seq_accu))
        bleu1, bleu2, bleu3, bleu4 = calculate_bleu_scores(reference, candidate)
        print(mean(avg_recall), mean(avg_precision), mean(avg_item_seq_accu), bleu1, bleu2, bleu3, bleu4)
        return mean(avg_recall), mean(avg_precision), mean(avg_item_seq_accu), bleu1, bleu2, bleu3, bleu4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Charades Training')
    parser.add_argument('--in-dir', default='charades/action_level_prediction/')
    parser.add_argument('--action-class-file', default='charades_action_class')
    args = parser.parse_args()
    evalutor = Evaluator(args)
    metrics = ["Recall", "Precision", "Item.Seq.Accu", "BLEU1", "BLEU2", "BLEU3", "BLEU4"]
    with open(f"{args.in_dir}/test_result_10.json","w") as f:
        json.dump(dict(zip(metrics, evalutor.evaluate())), f, cls=NumpyEncoder)
   
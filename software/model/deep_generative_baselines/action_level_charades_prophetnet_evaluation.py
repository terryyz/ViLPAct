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
        action_tgt_reference = pickle.load(open(f"{self.in_dir}/test_tgt_action.pkl","rb"))
        action_tgt_candidate = pickle.load(open(f"{self.in_dir}/test_pred.pkl","rb"))
        self.action_class = pickle.load(open(args.action_class_file,"rb"))
        self.ref = [[self.action_class[i] for i in action_ids] for action_ids in action_tgt_reference]
        self.cand = [tgt_actions[0][0].split(" ") for tgt_actions in action_tgt_candidate]

    def evaluate(self):
        avg_recall,avg_precision,avg_item_seq_accu = [],[],[]
        reference,candidate = [],[]
        print(len(self.ref), len(self.cand))
        for j, (ref, cand) in tqdm(enumerate(zip(self.ref, self.cand))):
            try:
                avg_recall.append(len(set(ref) & set(cand))/ len(set(ref)))
            except:
                avg_recall.append(0)
            try:
                avg_precision.append(len(set(ref) & set(cand))/ len(set(cand)))
            except:
                avg_precision.append(0)
            try:
                avg_item_seq_accu.append(sum([1 for i in range(min(len(ref),len(cand))) if cand[i] == ref[i]])/len(cand))
            except:
                avg_item_seq_accu.append(0)
            reference.append([ref])
            candidate.append(cand)
        bleu1, bleu2, bleu3, bleu4 = calculate_bleu_scores(reference, candidate)
        print(mean(avg_recall), mean(avg_precision), mean(avg_item_seq_accu), bleu1, bleu2, bleu3, bleu4)
        return mean(avg_recall), mean(avg_precision), mean(avg_item_seq_accu), bleu1, bleu2, bleu3, bleu4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Charades Training')
    parser.add_argument('--in-dir', default='charades/original_action_level_prediction/')
    parser.add_argument('--action-class-file', default='charades_action_class')
    args = parser.parse_args()
    evalutor = Evaluator(args)
    metrics = ["Recall", "Precision", "Item.Seq.Accu", "BLEU1", "BLEU2", "BLEU3", "BLEU4"]
    with open(f"{args.in_dir}/test_result.json","w") as f:
        json.dump(dict(zip(metrics, evalutor.evaluate())), f, cls=NumpyEncoder)
   
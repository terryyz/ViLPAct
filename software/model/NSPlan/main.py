import checkpoints
import json
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from models import create_i3d_model
from dataloader import get_dataset
from utils import tee
from knowledge_retrieval import KnowledgeRetriever, eval_seq
from answer_reasoner import answer_reasoner


def retrieve(opt):
    full_retrieve_file = opt.cache + '/full_retrieve_results_' + opt.mode
    # Save
    opt.save(opt.cache + '/config.json')
    tee.Tee(opt.cache + '/log.txt')
    # print(vars(opt))
    retriever = KnowledgeRetriever(opt)
    retriever.retrieve_init_video(full_retrieve_file + '.json', full_retrieve_file + '.csv',
                                              use_gt=opt.use_gt)


def reason(opt):
    task = opt.task
    full_retrieve_file = opt.cache + 'full_retrieve_results_' + opt.mode
    full_gt_file = opt.val_file

    kb_gt = pd.read_csv(opt.kb_seq)
    # kb_gt = kb_gt['action_flow_id'].to_list()
    kb_gt = kb_gt.action_flow_id.to_list()
    kb_seq = set()
    for gt in kb_gt:
        act_seq = ' '.join([act.split()[0] for act in gt.split(';')])
        kb_seq.add(act_seq)

    match_dir = opt.cache + 'reasoner'
    if not os.path.exists(match_dir):
        os.mkdir(match_dir)
    file =  '/reasoner_log.txt'
    tee.Tee(match_dir + file)

    # id2action:  id: action_text
    with open(opt.action_class, 'r') as act_f:
        action_cls_text = act_f.read().splitlines()
    action_cls_text = [action[5:] for action in action_cls_text]
    action2id = dict(zip(action_cls_text, range(len(action_cls_text))))

    # eval observed retrieved seq
    observed_retrieve_file = opt.cache + '/ob_retrieve_results_' + opt.mode
    if os.path.exists(observed_retrieve_file):
        observed_gt_file = opt.cache + '/observed_val_groundtruth.json'
        ob_seq_recall, ob_seq_pre, ob_exact_match, ob_pred_len, ob_mean_len, ob_bleu1, ob_bleu2, ob_bleu3, ob_bleu4, seq_hit, dist1, dist2 = eval_seq(
            observed_retrieve_file + '.json', observed_gt_file, kb_seq, opt.cache)
        ob_scores = {'seq_recall': ob_seq_recall, 'seq_pre': ob_seq_pre, 'exact_match': ob_exact_match,
                     'mean_len': ob_mean_len, 'pred_len': ob_pred_len, \
                     'bleu1': ob_bleu1, 'bleu2': ob_bleu2, 'bleu3': ob_bleu3, 'bleu4': ob_bleu4, 'dist1':dist1, 'dist2':dist2}
        checkpoints.score_file(ob_scores, filename="{}/eval_ob_seq.txt".format(opt.cache))
    #eval full retrieved seq
    full_seq_recall, full_seq_pre, full_exact_match, full_pred_len, full_mean_len, full_bleu1, full_bleu2, full_bleu3, full_bleu4, seq_hit, dist1, dist2 = eval_seq(
        full_retrieve_file + '.json', full_gt_file, kb_seq, opt.cache)
    full_scores = {'seq_recall': full_seq_recall, 'seq_pre': full_seq_pre, 'exact_match': full_exact_match,
                   'mean_len': full_mean_len, 'pred_len': full_pred_len, \
                   'bleu1': full_bleu1, 'bleu2': full_bleu2, 'bleu3': full_bleu3, 'bleu4': full_bleu4, 'seq_hit':seq_hit}
    checkpoints.score_file(full_scores, filename="{}/eval_full_seq.txt".format(opt.cache))
    # eval multi choice acc
    with open('{}/match_full_seq.txt'.format(opt.cache), 'w') as f:
        json.dump(answer_reasoner(full_retrieve_file + '.json', opt.qa_task_dir, opt.proto_feat_dir, action2id, task = task), f)


def main(opt):
    if opt.stage == 'retrieve':
        print('begin retrieve')
        retrieve(opt)

    elif opt.stage == 'reason':
        print('begin reason')
        reason(opt)



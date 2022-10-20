import numpy as np
import json
import os
from collections import defaultdict
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from text_match.seq_match import seq_match


def answer_reasoner(file, task_dir, proto_feat_dir, action2id, task='planning', vids=[], topk=10):
    if task == 'planning':
        task_file = 'planning_evaluate_data.json'
    print(task_dir + task_file)
    task_eval_data = json.load(open(task_dir + task_file, 'r'))
    match_methods = {'feat': ['TWED_embedding']}
    encodes = ['visual']
    acc_lst = defaultdict(int)
    multi_acc_lst = defaultdict(int)
    eval_num = 0
    mm = MinMaxScaler()
    if not len(vids):
        vids = task_eval_data.keys()
    with open(file, 'r') as f:
        pred_results = json.load(f)
        output = {}
        for result in tqdm(pred_results):
            vid = result['vid']
            if vid not in vids:
                continue
            eval_num += 1
            # topk 路径得分
            # result['topk'] = [fact for fact in result['topk'] if len(fact['act_seq']) > 0]
            k = min(topk, len(result['topk']))
            topk_scores = np.array([fact['score'] for fact in result['topk']])
            topk_scores = topk_scores[:k]
            if len(topk_scores) > 1:
                topk_scores = mm.fit_transform(np.array(topk_scores)[:, np.newaxis]).squeeze()
            topk_dis = np.exp(topk_scores) / sum(np.exp(topk_scores))
            #topk_dis = topk_dis[:k]
            # topk_dis = gumbel_softmax_sample(torch.from_numpy(topk_scores), 0.7).numpy()
            print('topk_dis:{}'.format(topk_dis))

            all_probs = {}
            gt = [action2id[act] for act in task_eval_data[vid]['answer'][1].split(',')]
            gt_index = task_eval_data[vid]['answer'][0]
            choice_num = len(task_eval_data[vid]['choices'])
            choices = [[action2id[act] for act in choice[1].split(',')] for choice in task_eval_data[vid]['choices']]

            # choice_eval_scores_lst:{'visual':{'TWED':[softmax(choice_prob)|path1], [softmax(choice_prob)|path2]...}}
            choice_eval_score_lst = dict([(encode, defaultdict(list)) for encode in encodes])

            for encode in encodes:
                if encode == 'id':
                    methods = match_methods['id']
                else:
                    methods = match_methods['feat']

                all_probs[encode] = dict(
                    [(method, []) for method in methods])

                # compute choice_eval_score_lst: {'visual':{'TWED':[[softmax(choice_prob)|path1], [softmax(choice_prob)|path2]]...}}
                # [[p(c1|p1), p(c2|p1), p(c3|p1)....],[p(c1|p2), p(c2|p2), p(c3|p2)...]]
                # for i in range(len(result['topk'])):
                for i in range(k):
                    # 针对每一个路径 result['topk'][i], 求和四个选项的匹配度
                    # choices_cond_path_score:{'visual':{'OT':[p(c1|path) p(c2|path), p(c3|path), p(c4|path)]}}
                    choices_cond_path_score = dict([(encode, defaultdict(list)) for encode in encodes])
                    pred_acts = [int(act.split()[0][1:]) for act in result['topk'][i]['act_seq']]
                    for choice in choices:
                        seq_match(pred_acts, choice, methods, encode=encode, position=False,
                                  feat_dir=proto_feat_dir, result=choices_cond_path_score)
                    for method in methods:
                        # [p(c1|path) p(c2|path), p(c3|path), p(c4|path)] normalization
                        choices_cond_path_score[encode][method] = np.exp(choices_cond_path_score[encode][method]) / sum(
                            np.exp(choices_cond_path_score[encode][method]))
                        for cj, cj_cond_pi in enumerate(choices_cond_path_score[encode][method]):
                            try:
                                assert np.isnan(cj_cond_pi) == False
                            except:
                                print(pred_acts, cj_cond_pi)
                                os.exit(-1)
                        # choice softmax distribution based on sequence i
                        choice_eval_score_lst[encode][method].append(choices_cond_path_score[encode][method])

                for method in methods:
                    # sum( p(choice|path)*p(path))
                    # sum(softmax(choice_prob)|path_i * path_i)
                    all_probs[encode][method] = np.sum(
                        np.array(choice_eval_score_lst[encode][method]) * topk_dis[:, np.newaxis], axis=0)
                    if encode + '_' + method not in acc_lst:
                        acc_lst[encode + '_' + method] = 0
                    if np.argmax(all_probs[encode][method]) == gt_index:
                        acc_lst[encode + '_' + method] += 1

                    # 多个最大值
                    max_prob_lst = [i for i, prob in enumerate(all_probs[encode][method]) if
                                    prob == max(all_probs[encode][method])]
                    if len(max_prob_lst) > 1:
                        multi_acc_lst[encode + '_' + method] += 1

            output[vid] = {'gt': all_probs[encode][method][gt_index],
                           'distracters': [prob for i, prob in enumerate(all_probs[encode][method]) if i != gt_index]}
            print('vid:{}, {}'.format(vid, output[vid]))
            print(acc_lst)

        for key, value in acc_lst.items():
            acc_lst[key] = value / eval_num
        for key, value in multi_acc_lst.items():
            multi_acc_lst[key] = value / eval_num

        print(eval_num, acc_lst)
        print('multi acc:{}'.format(multi_acc_lst))

        return output


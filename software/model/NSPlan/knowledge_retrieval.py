import joblib
import glob
import itertools
import random

from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import checkpoints
from models import create_i3d_model
from dataloader import get_dataset
from text_match.seq_match import seq_feat_match
from text_search.bm25 import BM25
from utils.metric_util import *
from action_detector import *
from tqdm import tqdm
import torch
import copy

def seed(manualseed):
    random.seed(manualseed)
    np.random.seed(manualseed)
    torch.manual_seed(manualseed)
    torch.cuda.manual_seed(manualseed)


INF = 1e12
BEGIN = '<BOS>'
END = '<EOS>'
seed(531)

class KnowledgeRetriever(object):
    def __init__(self, opt):
        self.opt = opt
        # load KB intention features
        with open(opt.test_intention_feat, 'rb') as f, \
                open(opt.kb_intention_feat, 'rb') as kb_f, \
                open(opt.kb_visual_featseq, 'rb') as kb_f1:
            self.test_intention_feat = pickle.load(f)
            self.kb_intention_feat = pickle.load(kb_f)
            self.kb_visual_featseq = pickle.load(kb_f1)

        #kb actions, kb_seqs, id2seq: uniqe seq
        self.id2action, self.id2seq, self.action2seq, self.kid2seqid, self.kid2featseq, \
        self.seqid2kids, self.kid2intentionfeat = self.createKB(opt)
        self.action2id = {act : idx for idx, act in self.id2action.items()}
        self.seq2id = {seq : idx for idx, (seq, count) in self.id2seq.items()}
        #with open('id2seq','w') as f, \
        #    open('kid2seqid', 'w') as f1:
        joblib.dump([(idx, seq) for idx, (seq, count) in self.id2seq.items()], './data/id2seq.json')
        joblib.dump(self.kid2seqid, './data/kid2seqid.json')


        # data
        self.test_loader, self.vocab = get_dataset(opt)
        self.observed_gt_file = '{}{}.json'.format(opt.cache, 'observed_groundtruth')
        self.kb_seq_file = '{}/{}.json'.format('./data', 'kb_seq.json')
        with open(self.kb_seq_file, 'w') as f:
            json.dump(list(self.seq2id.keys()), f)
        self.ov2a_matrix = self.ov2act()

        #feature extraction model
        i3d_model = create_i3d_model(opt)
        self.i3d_model = i3d_model
        self.i3d_model.eval()

        ## model to do the action recognition or localization
        if opt.mode == 'i3d_classifier':
            self.action_detector = I3DClassifier(opt=opt, model=self.i3d_model, vocab=self.vocab)

        elif opt.mode in ('Cosine', 'L2'):
            self.action_detector = FeatMatch(opt=opt, vocab=self.vocab)

        elif opt.mode == 'univl':
            self.action_detector = Univl(opt=opt)

        #retrival model
        corpus= [seq for (seq,count) in self.id2seq.values()]
        #BM 25
        self.bm25_model = BM25(corpus, delimiter=' ')
        # Metric
        self.act_recall_top1 = AverageMeter()
        self.act_recall_top5 = AverageMeter()
        self.obj_recall_top1 = AverageMeter()
        self.obj_recall_top5 = AverageMeter()
        self.verb_recall_top1 = AverageMeter()
        self.verb_recall_top5 = AverageMeter()
        self.act_pre_top1 = AverageMeter()
        self.act_pre_top5 = AverageMeter()
        # 8-frame chunk-leval
        self.chunk_act_recall_top1 = AverageMeter()
        self.chunk_act_recall_top5 = AverageMeter()
        self.chunk_act_pre_top1 = AverageMeter()
        self.chunk_act_pre_top5 = AverageMeter()

    #object,verb : act class
    def ov2act(self):
        ov2act_matrix = torch.LongTensor(self.opt.o_class, self.opt.v_class).fill_(-1)
        for key,value in self.vocab['c2ov'].items():
            ov2act_matrix[value[0], value[1]] = key
        return ov2act_matrix

    #initial_video -> retrieve from KB
    def retrieve_init_video(self, out_file, out_csv,  use_gt = False):
        """
        :param phase: train or val, use train data or val data
        :param out_file: topk retrival result
        :param kb_retrieve_method:  [start_action, inverted_index, BM25]
                observed_video   video_level action recognition   kb retrival
        :param use_gt: use the ground truth observed actions
        """
        opt = self.opt
        retrieve_results = []
        retrieve_future_results = []
        # true threshold
        threshold = []

        def part(x):
            return itertools.islice(x, int(len(x) * opt.val_size))

        loader = self.test_loader
        sample_size = opt.val_size
        if os.path.exists(opt.i3d_feature):
            self.vid2featseq = pickle.load(open(opt.i3d_feature, 'rb'))
        else:
            self.vid2featseq = {}

        # assert len(loader) == len(self.gt)
        total_examples = 0
        with torch.no_grad():
            for index, data in enumerate(tqdm(part(loader), total=int(len(loader) * sample_size))):
                ob_gt, future_gt, full_gt, rgb_segment_imgs, meta, observe_o, observe_s, observe_v, observe_a, frame_acts = data
                vid = meta['id'][0]
                segment_input = rgb_segment_imgs.cuda()  # [1,3,T,224,224]
                start_time = meta['time'][0].item()
                #end_time = start_time + segment_input.size(2) - 14
                full_gt = full_gt[0]
                ob_gt = ob_gt[0]
                total_examples += 1
                if os.path.exists(opt.i3d_feature):
                    init_feat_seq = self.vid2featseq[vid]
                    if init_feat_seq.shape[0] > 1:
                        q_feat = np.mean(init_feat_seq, axis=0, keepdims=True)  # [1,d]
                else:
                    ### feature extraction###
                    init_feat_seq, q_feat = self.feat_extraction(segment_input, vid)

                q_intention_feats = self.test_intention_feat[vid]
                print(q_intention_feats.shape)

                """
                stage1: observed video action recognition
                """
                # gt_act_indices_list
                if torch.sum(observe_a, dim=1) == 0:
                    gt_act_indices = []
                    print('the ground truth action is None:{},{}'.format(vid, start_time))
                else:
                    gt_act_indices = torch.nonzero(observe_a[0], as_tuple=True)[0].numpy().tolist()

                # video action recognition
                if self.opt.mode == 'univl':
                    topkacts, topk_seq_sim = self.action_detector.run(vid)
                else:
                    topkacts, topk_seq_sim = self.action_detector.run(segment_input, init_feat_seq, q_feat)

                act_rec_top1, act_rec_top5, obj_rec_top1, obj_rec_top5, v_rec_top1, v_rec_top5 = eval_act_recognition(topkacts, observe_a, self.opt.batch_size,
                                                                                                                        self.act_pre_top1, self.act_pre_top5, self.act_recall_top1, self.act_recall_top5,
                                                                                                                        self.vocab['c2ov'], observe_o, observe_v,
                                                                                                                        self.obj_recall_top1, self.obj_recall_top5, self.verb_recall_top1, self.verb_recall_top5)
                print('video:{} init_segment start:{} act_R@1:{}, act_R@5:{}，obj_R@1:{}, obj_R@5:{}, verb_R@1:{}, verb_R@5:{} \n gt:{}\n matching:{}\n'.format(vid, start_time, act_rec_top1.item(),
                                                                    act_rec_top5.item(), obj_rec_top1.item(),
                                                                    obj_rec_top5.item(), v_rec_top1.item(),
                                                                    v_rec_top5.item(),
                                                                    [self.id2action[aid] for aid in
                                                                     gt_act_indices],
                                                                    list(zip([self.id2action[aid] for aid in topkacts], topk_seq_sim))))


                """
                stage 2: kb retrival :: retrieve in KB, return sequences
                """
                full_gt = full_gt.numpy().tolist()
                ob_gt_acts = ob_gt.numpy().tolist()
                gt_ids = ' '.join([str(act) for act in full_gt]) #'106 107 108 109'
                for i,act in enumerate(topkacts):
                    if act in ob_gt_acts:
                        threshold.append(topk_seq_sim[i])
                gt = {'act_id': full_gt, 'des': [self.id2action[act] for act in full_gt]}
                topk_results = {'vid': vid, 'gt': gt, 'topk': []}

                print('observed gt:{}\t {}'.format(ob_gt_acts, [self.id2action[idx] for idx in ob_gt_acts]))
                print('full gt:{}\t {}'.format(gt['act_id'], gt['des']))


                # 1.set query
                # -if use observed action ground truth
                if use_gt:
                    topkacts = ob_gt_acts
                    topk_seq_sim = [1] * len(topkacts)
                    print('use gt acts:{}'.format(topkacts))
                else:
                    if opt.threshold:
                        if max(topk_seq_sim) > opt.threshold and min(topk_seq_sim) < opt.threshold:
                            topkacts = [act for idx, act in enumerate(topkacts) if
                                        topk_seq_sim[idx] > opt.threshold]
                        elif max(topk_seq_sim) < opt.threshold:
                            topkacts = topkacts[:3]
                    print('query acts greater than threshold:{} {}'.format(topkacts, [self.id2action[idx] for idx in topkacts]))
                # if use action recognition , can not obtain the order

                # 2.knowledge retrieval
                sorted_results = self.knowledge_retrival(self.opt.kb_retrieve_method, topkacts,
                                                         init_feat_seq, gt_ids, q_intention_feats)

                for result, score, actseq_des in sorted_results:
                    topk_results['topk'].append(
                        {'score': score, 'act_seq_des': actseq_des, 'act_seq': result})

                retrieve_results.append(topk_results)
                print('result: \t{}'.format(topk_results))
                if self.opt.filter_ob:
                    topk_future_results = copy.deepcopy(topk_results)
                    topk_future_results['topk'] = self.filter_ob(topk_results, topkacts)
                    retrieve_future_results.append(topk_future_results)
                    print('future acts result: \t{}'.format(topk_future_results['topk']))


            with open(out_file, 'w') as f:
                json.dump(retrieve_results, f)
            if opt.filter_ob:
                filter_file = os.path.join(opt.cache, '_'.join((str(opt.second_topk), opt.kb_retrieve_method, opt.rerank, 'filter_ob')))
                with open(filter_file, 'w') as f:
                    json.dump(retrieve_future_results, f)
           
            retrieve_df = pd.DataFrame(retrieve_results)
            retrieve_df.to_csv(out_csv)

            if not opt.i3d_feature:
                with open(opt.i3d_feature, 'wb') as f:
                    pickle.dump(self.vid2featseq, f)

            # segment_level action recognition scores
            scores = {'Act_P@1': self.act_pre_top1.avg, 'Act_P@5': self.act_pre_top5.avg,
                      'Act_R@1': self.act_recall_top1.avg, 'Act_R@5': self.act_recall_top5.avg,
                      'Obj_R@1': self.obj_recall_top1.avg, 'Obj_R@5': self.obj_recall_top5.avg,
                      'Verb_R@1': self.verb_recall_top1.avg, 'Verb_R@5': self.verb_recall_top5.avg,
                      'Clip_Act_P@1': self.chunk_act_pre_top1.avg, 'Clip_Act_P@5': self.chunk_act_pre_top5.avg,
                      'Clip_Act_R@1':self.chunk_act_recall_top1.avg, 'Clip_Act_R@5': self.chunk_act_recall_top5.avg}
            print(opt.cache)
            print(scores)
            print('mean of threshold:{}'.format(np.mean(threshold)))
            print('num of have predict:{}'.format(total_examples))
            if opt.verbose == 0:
                checkpoints.score_file(scores, filename="{}/eval_segment.txt".format(opt.cache))
            # sel retrival scores
            seq_recall, seq_pre, exact_match, pred_len, mean_len, bleu1, bleu2, bleu3, bleu4, seq_hit_rate, distinct1, distinct2 = eval_seq(
                out_file, self.opt.test_file, kb_seq=list(set(self.seq2id.keys())), out_dir=opt.cache)
            scores = {'num': total_examples, 'seq_recall': seq_recall, 'seq_pre': seq_pre, 'exact_match': exact_match, 'mean_len': mean_len,
                      'pred_len': pred_len, \
                      'bleu1': bleu1, 'bleu2': bleu2, 'bleu3': bleu3, 'bleu4': bleu4, 'seq_hit_rate':seq_hit_rate,
                      'distinct1':distinct1, 'distinct2':distinct2}

            if opt.verbose == 0:
                checkpoints.score_file(scores, filename="{}/eval_seq.txt".format(opt.cache))


    def feat_extraction(self, segment_input, vid):
        # feature extraction
        init_feat_seq = self.i3d_model(segment_input)  # [B, T,d]
        init_feat_seq = init_feat_seq[0].detach().cpu().numpy()  # [T, d]
        self.vid2featseq[vid] = init_feat_seq
        print('{}'.format(init_feat_seq.shape))
        if init_feat_seq.shape[0] > 1:
            q_feat = np.mean(init_feat_seq, axis=0, keepdims=True)  # [1,d]
        return init_feat_seq, q_feat


    # action recognition得到action 词袋
    def knowledge_retrival(self, kb_retrieve_method, topkacts, init_feat_seq, gt_ids, q_intention_feats = None):
        mm = MinMaxScaler()
        intention_scores = \
                self.video_intention_scores(q_intention_feats, kb_indices=list(self.kid2seqid.keys()),
                                            same_seq_label='max')
        seq_keys = self.id2seq.keys()
        if kb_retrieve_method == 'intention':
            scores = intention_scores
        # 使用BM25检索
        elif 'BM25' in kb_retrieve_method:
            query = ' '.join([str(act) for act in topkacts])
            bm_scores = self.bm25_model.getScores(query)
            # seq_id: seq_score
            scores = np.array(bm_scores)

            if 'intention' in kb_retrieve_method:
                scores = mm.fit_transform(scores[:, np.newaxis]).squeeze()
                #print('scores:{}'.format(scores[:]))
                # intention_scores = mm.fit_transform(intention_scores[:, np.newaxis]).squeeze()
                if self.opt.use_gt:
                    scores = 0.5 * scores + 0.5 * intention_scores
                else:
                    scores = 0.2 * scores + 0.8 * intention_scores
                i = 0
                for index in intention_scores.argsort()[-5:][::-1]:
                    print('top{}- kb sequence based on intention:{}{} {}\n'.format(i, index, self.id2seq[index],
                                                                                          [self.id2action[int(idx)] for
                                                                                           idx in self.id2seq[index][
                                                                                              0].split()]))
                    i += 1

        seq_scores = zip(seq_keys, scores)
        if self.opt.rerank:
                # BM25的结果取前50
                seq_scores = list(sorted(seq_scores, key=lambda x: x[1], reverse=True))[:self.opt.first_topn]
                sorted_seq_scores = self.rerank(self.opt.rerank, query, init_feat_seq, seq_scores, gt_ids, q_intention_feats = q_intention_feats)
        else:
                sorted_seq_scores = list(sorted(seq_scores, key=lambda x: x[1], reverse=True))
        seqids, scores = zip(*sorted_seq_scores)
        candidates = sorted_seq_scores[:self.opt.second_topk]
        # if gt_ids in kb
        if gt_ids in self.seq2id:
                gt_seqid = self.seq2id[gt_ids]
                top10_ids, top10_scores = zip(*candidates)
                print('gt_kb_score:{}, is in top10:{}'.format(scores[seqids.index(gt_seqid)] if gt_seqid in set(seqids) else 0,
                                                              gt_seqid in top10_ids))
        else:
                print('don\'t have the same seq in kb')


        seqs, path_scores = zip(*candidates)
        path_score_lst = np.exp(path_scores) / np.sum(np.exp(path_scores))
        act_seq_lst = [['c' + act.zfill(3) for act
                        in self.id2seq[seqid][0].split(' ')] for seqid in seqs]
        act_des_lst = [[self.id2action[int(aid)] for aid in self.id2seq[seqid][0].split(' ')] for seqid
                       in
                       seqs]
        sorted_results = zip(act_seq_lst, path_score_lst, act_des_lst)
        return sorted_results

    def rerank(self, rerank, query, init_feat_seq, seq_scores, gt_ids, partial_order = None, q_intention_feats = None):
        mm = MinMaxScaler()
        seqids, ori_scores = zip(*seq_scores)
        future_scores = list(
            map(lambda x: 0 if list(self.id2seq[x][0].split())[-1] in query.split() else 1, seqids))
        length_scores = list(
            map(lambda x: 1 if len(self.id2seq[x][0].split()) > len(query.split()) else 0, seqids))


        if rerank == 'future':
            scores = np.array(future_scores) + np.array(ori_scores)
            seq_scores = zip(seqids, scores)

        elif rerank == 'future_len':
            print('bm25:{}'.format(ori_scores[:10]))
            scores = np.array(ori_scores) + np.array(future_scores) + np.array(length_scores)
            print('future_len:{}'.format(scores[:10]))
            seq_scores = zip(seqids, scores)

        # video feature mean 进行rerank
        elif 'feat_match' in rerank:
            video_alignment = rerank
            video_alignment = video_alignment.replace('_future_len', '') if 'future_len' in rerank else video_alignment
            feat_match_scores = self.video_feat_scores(video_alignment, init_feat_seq, alignment = False, same_seq_label='add')
            video_feat_score = feat_match_scores[np.array(seqids)]
            scores = video_feat_score + ori_scores

            if 'future_len' in rerank:
                scores += np.array(future_scores) + np.array(length_scores)

            print('bm_scores:{}, feat_score:{}'.format(ori_scores[:10], video_feat_score[:10]))
            seq_scores = zip(seqids, scores)

        # visual feature alignment 进行 rerank
        elif 'DTW' in rerank or 'OTAM' in rerank or 'TWED' in rerank:
            video_alignment = rerank
            video_alignment = video_alignment.replace('_future_len', '') if 'future_len' in rerank else video_alignment
            video_alignment = video_alignment.replace('_bigram', '') if 'bigram' in rerank else video_alignment
            video_alignment = video_alignment.replace('_intention', '') if 'intention' in rerank else video_alignment
            kb_seq_ids = [kid for seqid in seqids for kid in self.seqid2kids[seqid]]
            assert len(kb_seq_ids) == len(set(kb_seq_ids))
            feat_alignment_scores = self.video_feat_scores(video_alignment, init_feat_seq,
                                                           kb_seq_ids,
                                                           alignment=True, same_seq_label='add')
            video_alignment_score = feat_alignment_scores[np.array(seqids)]
            #video_alignment_score = mm.fit_transform(video_alignment_score[:, np.newaxis]).squeeze()
            #ori_scores = mm.fit_transform(np.array(ori_scores)[:, np.newaxis]).squeeze()
            scores = video_alignment_score  +  np.array(ori_scores)

            if 'future_len' in rerank:
                if self.opt.use_gt:
                    scores += (np.array(future_scores) + np.array(length_scores)) / 2
                else:
                    scores += (np.array(future_scores) + np.array(length_scores)) / 5

            if 'intention' in rerank:
                intention_scores = \
                    self.video_intention_scores(q_intention_feats, kb_indices=kb_seq_ids,
                                                same_seq_label='max')
                scores = mm.fit_transform(scores[:, np.newaxis]).squeeze()
                # print('scores:{}'.format(scores[:]))
                # intention_scores = mm.fit_transform(intention_scores[:, np.newaxis]).squeeze()
                scores += intention_scores[np.array(seqids)]
                i = 0
                for index in intention_scores.argsort()[-5:][::-1]:
                    print('top{}- kb sequence based on intention:{}{} {}\n'.format(i, index, self.id2seq[index],
                                                                                   [self.id2action[int(idx)] for
                                                                                    idx in self.id2seq[index][
                                                                                        0].split()]))
                    i += 1


            print('bm_scores:{} \t feat_score:{}'.format(ori_scores[:10], video_alignment_score[:10]))
            print('most similarly video feat:{}\t {}'.format(self.id2seq[seqids[np.argmax(video_alignment_score)]],
                                                             [self.id2action[int(idx)] for idx in self.id2seq[seqids[np.argmax(video_alignment_score)]][0].split()]))
            print('most similarly bm25:{}\t {}'.format(self.id2seq[seqids[np.argmax(ori_scores)]],
                                                       [self.id2action[int(idx)] for idx in self.id2seq[seqids[np.argmax(ori_scores)]][0].split()]))
            print('most similarly mix :{}\t {}'.format(self.id2seq[seqids[np.argmax(scores)]],
                                                       [self.id2action[int(idx)] for idx in self.id2seq[seqids[np.argmax(scores)]][0].split()]))
            seq_scores = zip(seqids, scores)

        seq_scores = list(sorted(seq_scores, key=lambda x: x[1], reverse=True))

        return list(seq_scores)


    # if need filter the ob_query
    def filter_ob(self, result, ob_query):
        no_fact = 0
        new_result_topk = []
        for i, seq in enumerate(result['topk']):
            filtered_seq = []
            filtered_seq_des = []
            score = seq['score']
            for j, act in enumerate(seq['act_seq']):
                if int(act[1:]) not in ob_query:
                    filtered_seq.append(act)
                    filtered_seq_des.append(seq['act_seq_des'][j])
            if len(filtered_seq):
                new_result_topk.append({'score': score, 'act_seq': filtered_seq, 'act_seq_des': filtered_seq_des})
        if len(new_result_topk) == 0:
            new_result_topk.extend(result['topk'][:3])
            no_fact += 1
        print(no_fact)
        return new_result_topk

    #observed_video_visual_feat seq  ->  seq scores
    def video_feat_scores(self, match_method, init_feat_seq, kb_indices = [], alignment = False, same_seq_label = 'max'):
        """
        :param match_method:
        :param init_feat_seq: [chunk_num, dim]
        :param duration: init_feat_seq length
        :param kb_indices: candidates
        :param alignment: True or False(mean)
        :return: seq_score: [uniqe_seq_num, ]   #1969
        """
        opt = self.opt
        duration = init_feat_seq.shape[0]
        if alignment == False:
            q_feat = np.mean(init_feat_seq, axis=0, keepdims=True)
            kb_feat = np.zeros((len(self.kid2featseq), opt.extract_feat_dim)).astype('float32')
            for key in self.kid2featseq:
                # 根据初态的feat vector 个数来决定 每个kb中 序列的feat seq 取多少做平均
                kb_feat[key] = np.mean(self.kid2featseq[key][:duration], axis=0)

            # 若不进行时序的对齐，而是根据mean-pooling feature
            # faiss- kb seqs
            if not kb_indices:
                kb_indices = list(self.kid2seqid.keys())
            D, _I = self.featureSearch(q_feat, {'keys':kb_indices, 'feats':kb_feat[np.array(kb_indices)]}, len(kb_indices), opt.mode)
            if opt.mode == 'L2':
                sim = 1 / (1 + D)
            elif opt.mode == 'Cosine':
                sim = 0.5 + 0.5 * D

            seq_indices = [self.kid2seqid[idx] for idx in _I]
            seq_score = np.zeros(len(self.id2seq))
            # 可能有不同的kb_video对应同样的act sequence label，
            if same_seq_label == 'add':
                np.add.at(seq_score, seq_indices, sim)
            # 相同index的seq的得分取最大
            elif same_seq_label == 'max':
                np.maximum.at(seq_score, seq_indices, sim)
            print('most similarly kb sequence:{}'.format(self.id2seq[seq_indices[0]]))
            #sorted_seq = list(sorted(zip(range(max(seq_indices)+1), seq_score.tolist()), key=lambda x:x[1], reverse=True))
            return seq_score

        elif alignment == True:
            kb_candidates_scores = []
            for kb_index in kb_indices:
                #score = seq_feat_match(init_feat_seq, self.kid2featseq[kb_index][:duration], match_method=match_method, position=False)
                score = seq_feat_match(init_feat_seq, self.kid2featseq[kb_index], match_method=match_method, position=False)
                kb_candidates_scores.append(score)
            seq_indices = [self.kid2seqid[idx] for idx in kb_indices]
            seq_score = np.zeros(len(self.id2seq))
            # 可能有不同的kb_video对应同样的act sequence label，
            if same_seq_label == 'add':
                np.add.at(seq_score, seq_indices, kb_candidates_scores)
            # 相同index的seq的得分取最大
            elif same_seq_label == 'max':
                np.maximum.at(seq_score, seq_indices, kb_candidates_scores)
            i = 1
            for index in seq_score.argsort()[-5:][::-1]:
                print('top{}- kb sequence based on visual alignment:{}{} {}\n'.format(i, index, self.id2seq[index], [self.id2action[int(idx)] for idx in self.id2seq[index][0].split()]))
                i += 1
            # sorted_seq = list(sorted(zip(range(max(seq_indices)+1), seq_score.tolist()), key=lambda x:x[1], reverse=True))
            return seq_score


    def video_intention_scores(self, q_feats, kb_indices = [], same_seq_label='add'):
        """
        :param q_feats: [q_num, feat_dim]
        :param kb_indices: [kb index]
        :param same_seq_label: 'add', 'max'
        :return: seq_score
        """
        kb_intention1_feat = np.stack([value[0] for value in self.kid2intentionfeat.values()], axis=0).astype('float32') #[2402, 768]
        kb_intention2_feat = np.stack([value[1] for value in self.kid2intentionfeat.values()], axis=0).astype('float32')

        if kb_indices:
            kb_intention1_feat = kb_intention1_feat[np.array(kb_indices)]
            kb_intention2_feat = kb_intention2_feat[np.array(kb_indices)]

        sim1 = 0.5 + 0.5 * (1 - cdist(q_feats, kb_intention1_feat, 'cosine'))   #[2, kb_indices]
        sim2 = 0.5 + 0.5 * (1 - cdist(q_feats, kb_intention2_feat, 'cosine')) #[2, kb_indices]
        # 对于kb中每个候选， 在四对intention中取匹配的最好的intention得分
        sim_pair = np.concatenate((sim1, sim2), axis=0)  # [4, kb_indices]F

        #sim = np.max(sim_pair, axis=0) #[kb_indices]
        sim = np.mean(sim_pair, axis=0)
        seq_indices = [self.kid2seqid[idx] for idx in kb_indices]
        seq_score = np.zeros(len(self.id2seq))
        # 可能有不同的kb_video对应同样的act sequence label，
        if same_seq_label == 'add':
            np.add.at(seq_score, seq_indices, sim)
        # 相同index的seq的得分取最大
        elif same_seq_label == 'max':
            np.maximum.at(seq_score, seq_indices, sim)
        # sorted_seq = list(sorted(zip(range(max(seq_indices)+1), seq_score.tolist()), key=lambda x:x[1], reverse=True))
        return seq_score

    # if use prototype feature
    def featureSearch(self,  q_feats, candidates, topk, mode):
        #mode: IP:cosine， L2:L2距离
        if mode == "IP" or mode == "Cosine":
            index = faiss.IndexFlatIP(candidates['feats'].shape[1])
            if mode == "Cosine":
                faiss.normalize_L2(candidates['feats'])
                faiss.normalize_L2(q_feats)

        elif mode == "L2" :
            index = faiss.IndexFlatL2(candidates['feats'].shape[1])

        index.add(candidates['feats'])
        D, _I = index.search(q_feats, topk)  #[q_num, topk]
        acts = [candidates['keys'][index] for index in _I[0]]
        return D[0], acts


    def createKB(self, args):
        """
        :param args:
        :return: id2action:dict, key:int class of action, value:action description
        :return id2seq: dict, key:int index of action sequence(1969, unique), value: action sequence description
        :return action2seq:dict, key:int class of action, value:int index of action sequence
        :return kid2seqid:dict, key:int index of kb sequence(2401, have duplicate), value: index of action sequence
        :return kid2featseq, key:int index of kb sequence , value: visual feature sequence of this sequence
        :return seqid2kids, key: index of action sequence, value: indices of kb sequence
        """
        kb_data = pd.read_csv(args.kb_seq)
        print('kb data :{}'.format(len(kb_data)))

        #id2action:  id: action_text
        with open(args.action_class, 'r') as act_f:
                action_cls_text = act_f.read().splitlines()
        action_cls_text = [action[5:] for action in action_cls_text]
        action2id = dict(zip(action_cls_text, range(len(action_cls_text))))
        id2action = {index: action for action, index in action2id.items()}

        # inverted index
        action2seq = defaultdict(set)
        seq_ids_lst = list(map(lambda x:' '.join([str(int(act.split()[0][1:])) for act in x.split(';')]), kb_data['action_flow_id'].to_list()))
        seq_counter = Counter(seq_ids_lst)
        #unique seq id
        id2seq = dict(zip(range(len(seq_counter)), seq_counter.items())) # {seqid: (act_ids,count)}
        seq2id = {seq:idx for idx,(seq,count) in id2seq.items()}
        kid2seqid = {}
        kid2featseq = {}
        kid2intentionfeat = {}

        # seqid -> [kb data id]
        seqid2kids = defaultdict(list)
        print('number of seq:{}'.format(len(seq_counter)))

        for i, row in kb_data.iterrows():
            action_seq = row['action_flow_id'].split(';')
            vid = row['id']
            seq = ' '.join([str(int(act.split()[0][1:])) for act in action_seq])  # '106 108 107'
            seq_id = seq2id[seq]
            kid2seqid[i] = seq_id
            seqid2kids[seq_id].append(i)
            #intention_feat = (intention1_feat, intention2_feat)
            kid2intentionfeat[i] = self.kb_intention_feat[vid]
            kid2featseq[i] = self.kb_visual_featseq[vid]

        #print('the candidates for each action:{}'.format(next_acts))
        return id2action, id2seq, action2seq, kid2seqid, kid2featseq, seqid2kids, kid2intentionfeat





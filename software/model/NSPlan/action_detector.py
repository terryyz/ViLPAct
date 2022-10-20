import torch
import numpy as np
import faiss
import pickle



# if use video clip feature
class ActionDetector:
    def __init__(self, opt, model, vocab):
        self.model = model
        self.vocab = vocab
        self.opt = opt
    def run(self, input, feat_seq, feat):
        """
        :param input: input video
        :param feat_seq: snippet-level feature of input video
        :param feat: video-level feature of input video
        """
        logit = self.model(input)  # [B,C,T]
        prob = torch.nn.Sigmoid()(torch.max(logit, dim=2)[0])
        print('prob:{}'.format(prob.size))
        topk_prob, topk_indices = prob.topk(self.opt.act_topk, 1, True, True)  # [1,k]
        # topk_indices = (prob > 0.5).nonzero()
        # topk_seq_sim = torch.index_select(prob, 1, indices.squeeze(0))  # [1, |C|]
        topkacts = topk_indices.squeeze(0).detach().cpu().numpy().tolist()  # [topk,]
        topk_seq_sim = topk_prob.squeeze(0).detach().cpu().numpy().tolist()
        return topkacts, topk_seq_sim

class I3DClassifier(ActionDetector):
    def run(self, input, feat_seq, feat):
        logit, _ = self.model(input, False)  # [B,C,T]
        prob = torch.nn.Sigmoid()(torch.max(logit, dim=2)[0])
        print('prob:{}'.format(prob.size))
        topk_prob, topk_indices = prob.topk(self.opt.act_topk, 1, True, True)  # [1,k]
        # topk_indices = (prob > 0.5).nonzero()
        # topk_seq_sim = torch.index_select(prob, 1, indices.squeeze(0))  # [1, |C|]
        topkacts = topk_indices.squeeze(0).detach().cpu().numpy().tolist()  # [topk,]
        topk_seq_sim = topk_prob.squeeze(0).detach().cpu().numpy().tolist()
        return topkacts, topk_seq_sim

class Univl:
    def __init__(self, opt):
        dict_path = opt.univl_query
        with open(dict_path, 'rb') as f:
            self.vid2query = pickle.load(f)
    def run(self, vid):
        ob_action_seq = self.vid2query[vid]
        topkacts = list(set([int(act[1:]) for act in ob_action_seq]))
        topk_seq_sim = [1.0 for _ in range(len(topkacts))]
        return topkacts, topk_seq_sim

class FeatMatch(ActionDetector):
    def __init__(self, opt, vocab):
        super(FeatMatch, self).__init__(opt=opt, model=None, vocab=vocab)
        self.mode = self.opt.mode
        self.feat_dim = self.opt.extract_feat_dim
        # action features
        self.feats = self.load_kb_actFeature(self.opt.visual_proto_path) #[157,d]
        assert self.feats.shape[0] == len(self.vocab['id2action'])
        #self.feats_var = self.load_kb_actFeature(opt.feat_var_path, opt.kb_feature) #[157,d]


    def run(self, input, feat_seq, q_feat):
        candidates = {'keys': list(self.vocab['id2action'].keys()),
                      'feats': self.feats}  # (action_indices, action_feats)
        D, topkacts = self.featureSearch(q_feat, candidates, self.opt.act_topk, self.mode)  # [topk,]
        if self.mode == 'L2':
            topk_seq_sim = 1 / (1 + D)
        elif self.mode == 'Cosine':
            topk_seq_sim = 0.5 + 0.5 * D

        return topkacts, topk_seq_sim

        # if use prototype feature

    def featureSearch(self, q_feats, candidates, topk, mode):
        # mode: IP:cosine， L2:L2距离
        if mode == "IP" or mode == "Cosine":
            index = faiss.IndexFlatIP(candidates['feats'].shape[1])
            if mode == "Cosine":
                faiss.normalize_L2(candidates['feats'])
                faiss.normalize_L2(q_feats)

        elif mode == "L2":
            index = faiss.IndexFlatL2(candidates['feats'].shape[1])

        index.add(candidates['feats'])
        D, _I = index.search(q_feats, topk)  # [q_num, topk]
        acts = [candidates['keys'][index] for index in _I[0]]
        return D[0], acts


    def load_kb_actFeature(self, prototype_feat_path):
        # each action has a feature vector
        feats = np.stack(pickle.load(open(prototype_feat_path, 'rb')).values(), axis=0)
        #feats = np.load(prototype_feat_path)  # [157,args.extract_feat_dim]
        print(feats.shape)
        assert feats.shape == (157, self.opt.extract_feat_dim)
        return feats


""" Define and parse commandline arguments """
import argparse
import os
import six
import json
import copy
import pprint


os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    @classmethod
    def from_dict(cls, dict_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = Config()
        for (key, value) in six.iteritems(dict_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `AlbertConfig` from a json file of parameters."""
        with open(json_file, 'r') as f:
            text = json.load(f)
        return cls.from_dict(text)

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def save(self, out_file):
        output = self.to_dict()
        with open(out_file,'w') as f:
            json.dump(output,f)

def read_config(path):
    return Config.from_json_file(path)

def get_config(**optional_kwargs):
    print('parsing arguments')
    parser = argparse.ArgumentParser(description='PyTorch Charades Training')
    # dataset
    parser.add_argument('--rgb-data', metavar='DIR', default='/hdd1/liaoyaqing/Charades_v1_rgb/',
                        help='path to dataset')
    parser.add_argument('--dataset', metavar='DIR', default='ViLPAct',
                        help='name of dataset under datasets/')
    parser.add_argument('--init_sample', default=False,
                        help='if sample the frames of init video')
    parser.add_argument('--task',  default='planning',
                        help='evaluation task type')
    parser.add_argument('--train-file', default='./data/gt_act_seq/train_planning_gt_seq.csv')
    parser.add_argument('--val-file', default='./data/gt_act_seq/val_planning_gt_seq.csv')
    parser.add_argument('--test-file', default='./data/gt_act_seq/test_planning_gt_seq.csv')
    parser.add_argument('--rgb-pretrained-weights', default='./pretrain_models/rgb_charades.pt')
    parser.add_argument('--i3d-feature', default = './data/test_features/test_video_features.pickle')
    parser.add_argument('--test-intention-feat', default='./data/test_features/test_intent_feat.pkl')

    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 1)')
    parser.add_argument('--inputsize', default=224, type=int)
    parser.add_argument('--extract-feat-dim', default=1024, type=int)
    parser.add_argument('--extract-text-dim', default=768, type=int)
    parser.add_argument('--manual-seed', default=0, type=int)
    parser.add_argument('--train-size', default=1, type=float)
    parser.add_argument('--val-size', default=1, type=float)
    parser.add_argument('--cache-dir', default='')
    parser.add_argument('--name', default='')
    parser.add_argument('--s-class', default=16, type=int)
    parser.add_argument('--o-class', default=38, type=int)
    parser.add_argument('--v-class', default=33, type=int)
    parser.add_argument('--max-chunk', default=200, type=int)
    parser.add_argument('--stack', default=10, type=int)
    parser.add_argument('--gap', default=4, type=int)
    parser.add_argument('--workers', default=1, type=int)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')

    #kb
    parser.add_argument('--kb-seq', default='./data/KB/video_activity.csv')
    parser.add_argument('--action_class', default='/hdd1/liaoyaqing/charades/Charades_v1_classes.txt')
    parser.add_argument('--visual-proto-path', default = './data/KB/features/aid2visualprototype.pickle')
    parser.add_argument('--kb-intention-feat', default='./data/KB/features/vid2intentfeat.pkl')
    parser.add_argument('--kb-visual-featseq', default='./data/KB/features/vid2featseq.pickle')

    #KB retrieval
    parser.add_argument('--mode', default='L2', choices=['L2', 'i3d_classifier', 'Cosine', 'SlowFast'], help='action recognition or detection')
    parser.add_argument('--act_topk', default=5, help='number of mumximum BM25 query')
    parser.add_argument('--threshold', default='',  help='threshold of action recognition')
    parser.add_argument('--first_topn', default=50, help='number of first retrieval results')
    parser.add_argument('--second_topk', default=10, help='number of second retrieval results')
    parser.add_argument('--kb-retrieve-method', default='BM25_intention', choices=['BM25', 'BM25_intention'], help='Knowlege initial retrieval')
    parser.add_argument('--rerank', default='', choices=['feat_match', 'DTW_embedding', 'OTAM_embedding', 'future',
                                                         'future_len'],
                        help='retrieval rerank method')
    parser.add_argument('--filter-ob', default=True, help='filter the predictive ob in retrieved results')

    # Reasoner
    parser.add_argument('--qa-task-dir', default='./data/qa_evaluation/')
    parser.add_argument('--proto-feat-dir', default='./data/prototype_feature/')


    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--stage','-s', default='retrieve', choices=['retrieve', 'reason'])
    parser.add_argument('--debug', action='store_true')


    kwargs = parser.parse_args()
    kwargs.distributed = kwargs.world_size > 1
    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    kwargs['predict_triplets'] = kwargs['cache_dir'] +  'predict_triplets.p'
    kwargs['cache'] =kwargs['cache_dir'] + kwargs['name'] + '/'
    if not os.path.exists(kwargs['cache']):
        os.makedirs(kwargs['cache'])

    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()

    # Save
    config.save('config.txt')

    # Load
    loaded_config = read_config('config.txt')

    assert config.__dict__ == loaded_config.__dict__

    import os

    os.remove('config.txt')

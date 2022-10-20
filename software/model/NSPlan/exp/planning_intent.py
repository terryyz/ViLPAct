import os
import torch
import sys
sys.path.insert(0, '.')
from main import main
from opts import get_config, read_config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print('use {} gpu'.format(torch.cuda.device_count())
)


update = {
    #'name': __file__.split('/')[-1].split('.')[0],  # name is filename
    'cache_dir': './results/',
    'task': 'planning',
    'init_sample': False,
    'dataset': 'ViLPAct',

    'mode':'intent', #[L2, i3d_classifier, Cosine, Univl,intent],
    'univl_query':'/hdd1/liaoyaqing/charades/univl_query.pkl',
    'kb_retrieve_method': 'intention',  # [BM25, BM25_intention,intention]
    'rerank': '', # [DTW_embedding, OTAM_embedding, future, future_len, OTAM_embedding_future_len, feat_match]
    'use_gt':False,
    'filter_ob':False,
    'threshold': '',
    'shuffle': False,

    'batch_size': 1,
    'act_topk': 5,
    'first_topn': 50,
    'second_topk': 10,
    'verbose':0,
}  #


update['name'] = '_'.join(
    ('top_' + str(update['second_topk']),  update['kb_retrieve_method']))
if update['use_gt']:
    update['name'] += '_gt'
if update['threshold']:
    update['name'] += '_' + str(update['threshold'])
if update['rerank']:
    update['name'] = update['name'] + '_' + update['rerank']



if __name__ == '__main__':
    #config_file = '/hdd1/liaoyaqing/knowledge_retrieve_act_forecast/20%_L2_beam10_prototype_BM25_intention_gt_OTAM_embedding_future_len/config.json'
    config_file = ''

    if config_file:
        print('from config')
        opt = read_config(config_file)
        print(opt.__dict__)
        print(opt.stage)
    else:
        print('from update')
        print(update)
        opt = get_config(**update)

    main(opt)

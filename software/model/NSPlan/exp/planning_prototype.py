import os
import torch
import sys
sys.path.insert(0, '.')
from main import main
from opts import get_config, read_config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
print('use {} gpu'.format(torch.cuda.device_count())
)


update = {
    #'name': __file__.split('/')[-1].split('.')[0],  # name is filename
    'cache_dir': './results/',
    'task': 'planning',
    'init_sample': False,
    'dataset': 'ViLPAct',

    'mode':'Cosine', #[L2, i3d_classifier, Cosine, Univl],
    'kb_retrieve_method': 'BM25',  # [BM25, BM25_intention, intention]
    'rerank': 'OTAM_embedding_future_len', #[DTW_embedding, OTAM_embedding, future, future_len, OTAM_embedding_future_len, feat_match]
    'use_gt':False,
    'filter_ob':True,
    'threshold': 0.85,
    'shuffle': False,

    'batch_size': 1,
    'act_topk': 5,
    'first_topn': 50,
    'second_topk': 10,
    'verbose':0}  #


update['val_task_data'] = '{}/{}_{}_{}.pkl'.format(update['cache_dir'],
                                             update['dataset'], 'act', update['task'])
update['name'] = '_'.join(
    (update['init_method'], update['mode'], 'beam' + str(update['beam_width']), update['kb_retrieve_method']))
if update['use_gt']:
    update['name'] += '_gt'
if update['threshold']:
    update['name'] += '_' + str(update['threshold'])
if update['rerank']:
    update['name'] = update['name'] + '_' + update['rerank']


if __name__ == '__main__':
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

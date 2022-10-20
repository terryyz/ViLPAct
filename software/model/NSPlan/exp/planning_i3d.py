import os
import torch
import sys
sys.path.insert(0, '.')
from main import main
from opts import get_config, read_config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
print('use {} gpu'.format(torch.cuda.device_count())
)


update = {
    #'name': __file__.split('/')[-1].split('.')[0],  # name is filename
    'cache_dir': './results/',
    'task': 'planning',
    'init_sample': False,
    'dataset': 'ViLPAct',

    'mode':'i3d_classifier', #[L2, i3d_classifier, Cosine, Univl],
    'kb_retrieve_method':'BM25',  #[BM25, BM25_intention]
    'rerank': 'OTAM_embedding_future_len',   #[DTW_embedding, OTAM_embedding, future, future_len, bigram, feat_match]
    'use_gt':False,
    'filter_ob':True,
    'threshold': 0.3,
    'shuffle': False,

    'batch_size': 1,
    'first_topn': 50,
    'act_topk': 5,
    'second_topk': 10,
    'verbose':0}  #


update['name'] = '_'.join(
    ('top' + str(update['second_topk']),  update['mode'], update['kb_retrieve_method']))
if update['use_gt']:
    update['name'] += '_gt'
if update['threshold']:
    update['name'] += '_' + str(update['threshold'])
if update['rerank']:
    update['name'] = update['name'] + '_' + update['rerank']




if __name__ == '__main__':
    config_file = ''
    #config_file = "/hdd1/liaoyaqing/knowledge_retrieve_act_forecast/baseline_planning_full/config.json"

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

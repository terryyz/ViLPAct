""" Initilize the datasets module
    New datasets can be added with python scripts under datasets/
"""
import torch
import torch.utils.data
import torch.utils.data.distributed
import importlib
from dataloader.utils import make_vocab
from opts import get_config
def get_dataset(args):
    dataset = importlib.import_module('.'+args.dataset, package='dataloader')
    test_dataset = dataset.get(args)
    obj2id, verb2id, s_lab2int, act2id, c2ov = make_vocab()
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    id2act = {index: action for action, index in act2id.items()}
    vocab = {'action2id': act2id, 'id2action': id2act, 'c2ov':c2ov, 'obj2id':obj2id, 'verb2id':verb2id}
    return test_loader, vocab

if __name__ == '__main__':
    args = get_config()
    print(args.__dict__)
    test_data, vocab = get_dataset(args)
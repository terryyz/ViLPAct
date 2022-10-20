""" Defines functions used for checkpointing models and storing model scores """
import os
import torch
import shutil
from collections import OrderedDict


def ordered_load_state(model, chkpoint):
    """ 
        Wrapping the model with parallel/dataparallel seems to
        change the variable names for the states
        This attempts to load normally and otherwise aligns the labels
        of the two statese and tries again.
    """
    try:
        model.load_state_dict(chkpoint)
        print('load {} from {}'.format(locals()[model], chkpoint))
    except :  # assume order is the same, and use new labels
        print('keys do not match model, trying to align')
        modelkeys = model.state_dict().keys()
        fixed = OrderedDict([(z,y) 
                             for (x,y),z in zip(chkpoint.items(), modelkeys)])
        model.load_state_dict(fixed)


def load_pretrain(args, base_model, logits_model, base_optimizer=None, logits_optimizer=None):
    if args.pretrain_resume:
        if os.path.isfile(args.pretrain_resume):
            print("=> loading pretrain checkpoint '{}'".format(args.pretrain_resume))
            if args.distributed:
               loc = 'cuda: {}'.format(args.local_rank)
               chkpoint = torch.load(args.pretrain_resume, map_location=loc)
            else:
                chkpoint = torch.load(args.pretrain_resume)
            if isinstance(chkpoint, dict) and 'base_state_dict' in chkpoint:
                mAP = chkpoint['mAP']
                ordered_load_state(base_model, chkpoint['base_state_dict'])
                ordered_load_state(logits_model, chkpoint['logits_state_dict'])
                if base_optimizer is not None and logits_optimizer is not None:
                    args.start_epoch = chkpoint['epoch']
                    base_optimizer.load_state_dict(chkpoint['base_optimizer'])
                    logits_optimizer.load_state_dict(chkpoint['logits_optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.pretrain_resume, chkpoint['epoch']))
                return mAP
            else:
                ordered_load_state(base_model, chkpoint)
                ordered_load_state(logits_model, chkpoint)
                print("=> loaded checkpoint '{}' (just weights)"
                      .format(args.pretrain_resume))
                return 0
        else:
            #raise ValueError("no checkpoint found at '{}'".format(args.resume))
            #raise ValueError("no checkpoint found at '{}'".format(args.resume))
            print("=> no checkpoint found, starting from scratch: '{}'".format(args.pretrain_resume))
    return 0

def load(args, model, optimizer):
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.distributed:
               loc = 'cuda: {}'.format(args.local_rank)
               chkpoint = torch.load(args.resume, map_location=loc)
            else:
                chkpoint = torch.load(args.resume)
            if isinstance(chkpoint, dict) and 'state_dict' in chkpoint:
                args.start_epoch = chkpoint['epoch']
                best_loss = chkpoint['best_loss']
                ordered_load_state(model, chkpoint['state_dict'])
                optimizer.load_state_dict(chkpoint['optimizer'])
                optimizer = chkpoint['optimizer']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, chkpoint['epoch']))
                return best_loss, optimizer
            else:
                ordered_load_state(model, chkpoint)
                print("=> loaded checkpoint '{}' (just weights)"
                      .format(args.resume))
                return 0
        else:
            #raise ValueError("no checkpoint found at '{}'".format(args.resume))
            #raise ValueError("no checkpoint found at '{}'".format(args.resume))
            print("=> no checkpoint found, starting from scratch: '{}'".format(args.resume))
    return 0

def score_file(scores, filename):
    with open(filename, 'w') as f:
        for key, val in sorted(scores.items()):
            f.write('{} {}\n'.format(key, val))


def save_pretrain(epoch, args, base_model, logits_model, base_optimizer, logits_optimizer, is_best, scores):
    state = {
        'epoch': epoch + 1,
        'rgb_arch': args.rgb_arch,
        'base_state_dict': base_model.state_dict(),
        'logits_state_dict': logits_model.state_dict(),
        'mAP': scores['mAP'],
        'base_optimizer': base_optimizer.state_dict(),
        'logits_optimizer': logits_optimizer.state_dict(),
    }
    filename = "{}/model.pth.tar".format(args.cache)
    score_file(scores, "{}/model_{:03d}.txt".format(args.cache, epoch+1))
    torch.save(state, filename)
    if is_best:
        bestname = "{}/model_best.pth.tar".format(args.cache)
        score_file(scores, "{}/model_best.txt".format(args.cache, epoch+1))
        shutil.copyfile(filename, bestname)

def save(epoch, args, model, optimizer, is_best, scores):
    state = {
        'epoch': epoch + 1,
        'rgb_arch': args.rgb_arch,
        'state_dict': model.state_dict(),
        'best_loss': scores['best_loss'],
        'optimizer': optimizer
    }
    filename = "{}/model.pth.tar".format(args.cache)
    score_file(scores, "{}/model_{:03d}.txt".format(args.cache, epoch+1))
    torch.save(state, filename)
    if is_best:
        bestname = "{}/model_best.pth.tar".format(args.cache)
        score_file(scores, "{}/model_best.txt".format(args.cache, epoch+1))
        shutil.copyfile(filename, bestname)

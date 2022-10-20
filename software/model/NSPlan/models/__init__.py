"""
Initialize the model module
New models can be defined by adding scripts under models/
"""
import torch
import torch.nn.parallel
import torch.distributed as dist




def create_i3d_model(args):
    pretrained_weights = args.rgb_pretrained_weights
    from models.i3d import InceptionI3d
    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(157)
    if not pretrained_weights == '':
        print('loading pretrained-weights from {}'.format(pretrained_weights))
        model.load_state_dict(torch.load(pretrained_weights))


    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
        """
        dist.init_process_group("nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        """
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if hasattr(model, 'features'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define optimizer
    #optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)
    return model
""""""


    



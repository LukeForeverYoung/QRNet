import argparse
import datetime
from functools import partial
import json
import random
import time
import math

import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import utils.misc as utils
from models import build_model
from datasets import build_dataset
from engine import train_one_epoch, validate
from icecream import ic

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=1e-5, type=float)
    parser.add_argument('--lr_visu_swin', default=1e-5, type=float)
    parser.add_argument('--lr_visu_fpn', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=180, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--clip_max_norm', default=0., type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr_scheduler', default='step', type=str)
    parser.add_argument('--lr_drop', default=80, type=int)
    
    # Augmentation options
    parser.add_argument('--aug_blur', action='store_true',
                        help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true',
                        help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true',
                        help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true',
                        help="If true, use random translate augmentation")
    
    # Model parameters
    # DETR parameters
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=0, type=int, # Edit from TransVG
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--imsize', default=640, type=int, help='image size')
    parser.add_argument('--emb_size', default=512, type=int,
                        help='fusion module embedding dimensions')

    # Transformers in two branches
    parser.add_argument('--bert_enc_num', default=12, type=int)

    # Vision-Language Transformer
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='./ln_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='referit', type=str,
                        help='referit/flickr/unc/unc+/gref')
    parser.add_argument('--max_query_len', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    
    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--bert_model', default='./checkpoints/bert-base-uncased', type=str, help='bert model')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_true',
                        help="If true, use amp training")
    parser.add_argument('--init_model_from', default=None,type=str,
                        help="Use a checkpoint to init model")
    parser.add_argument('--soft_fpn', default='NoFpnSoftDownSample',type=str)
    parser.add_argument('--disable_spatial', action='store_true',
                        help="If true, use amp training")
    parser.add_argument('--disable_channel', action='store_true',
                        help="If true, use amp training")
    parser.add_argument('--swin_checkpoint', default='checkpoints/mask_rcnn_swin_small_patch4_window7.pth',type=str)
    return parser


def get_model_param_list(model,model_without_ddp,args):
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    visu_swin_rule=lambda n,p: (("visumodel" in n) and ("backbone" in n) and ('qdatt' not in n) and p.requires_grad)
    visu_dqa_rule=lambda n,p: (("visumodel" in n) and ("backbone" in n) and ('qdatt' in n) and p.requires_grad)
    visu_fpn_rule = lambda n,p: (("visumodel" in n) and ("backbone" not in n) and ('fpn_down' not in n) and p.requires_grad)
    visu_fpn_down_rule=lambda n,p: (("visumodel" in n) and ("backbone" not in n) and ('fpn_down' in n) and p.requires_grad)
    text_tra_rule = lambda n,p: (("textmodel" in n) and p.requires_grad)
    pruning_rule = lambda n,p: (("pruning_model" in n) and p.requires_grad)
    rest_rule = lambda n,p: (("visumodel" not in n) and ("textmodel" not in n) and ('pruning_model' not in n) and p.requires_grad)
    all_rule=[visu_swin_rule,visu_dqa_rule,visu_fpn_rule,visu_fpn_down_rule,text_tra_rule,pruning_rule,rest_rule]
    visu_swin_param = [p for n, p in model_without_ddp.named_parameters() if visu_swin_rule(n,p)]
    visu_dqa_param = [p for n, p in model_without_ddp.named_parameters() if visu_dqa_rule(n,p)] 
    visu_fpn_param = [p for n, p in model_without_ddp.named_parameters() if visu_fpn_rule(n,p)]
    text_tra_param = [p for n, p in model_without_ddp.named_parameters() if text_tra_rule(n,p)]
    visu_fpn_down_param = [p for n, p in model_without_ddp.named_parameters() if visu_fpn_down_rule(n,p)]
    rest_param = [p for n, p in model_without_ddp.named_parameters() if rest_rule(n,p)] 
    param_list = [{"params": rest_param,            "lr":args.lr},
                   {"params": visu_swin_param,       "lr": args.lr_visu_swin},
                   {"params": visu_dqa_param,  "lr":args.lr},
                   {"params": visu_fpn_param,       "lr": args.lr_visu_fpn},
                   {"params": visu_fpn_down_param,  "lr":args.lr},
                   {"params": text_tra_param,       "lr": args.lr_bert},
                   ]
    for pl in param_list:
        params=pl['params']
        p_num=sum(p.numel() for p in params)

    return param_list,n_parameters
def main(args):
    args.use_channel=not args.disable_channel
    args.use_spatial=not args.disable_spatial
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # build model
    model = build_model(args)
    model.to(device)
    if args.init_model_from is not None:
        init_checkpoint=torch.load(args.init_model_from)
        model.load_state_dict(init_checkpoint['model'])
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    param_list,n_parameters=get_model_param_list(model,model_without_ddp,args)
    
    # using RMSProp or AdamW
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError('Lr scheduler type not supportted ')
    
    # using polynomial lr scheduler or half decay every 10 epochs or step
    if args.lr_scheduler == 'poly':
        lr_func = lambda epoch: (1 - epoch / args.epochs) ** args.lr_power
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'halfdecay':
        lr_func = lambda epoch: 0.5 ** (epoch // (args.epochs // 10))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'cosine':
        lr_func = lambda epoch: 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    else:
        raise ValueError('Lr scheduler type not supportted ')

    # build dataset
    dataset_train = build_dataset('train', args)
    dataset_val   = build_dataset('val', args)
    ## note certain dataset does not have 'test' set:
    ## 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    # dataset_test  = build_dataset('test', args)
    
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val   = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val   = torch.utils.data.SequentialSampler(dataset_val)
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=partial(utils.collate_fn), num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn_val, num_workers=args.num_workers)

    

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(str(args) + "\n")

   
    print("Start training")
    start_time = time.time()
    best_accu = 0
    # Creates a GradScaler once at the beginning of training.
    from torch.cuda.amp import GradScaler
    scaler = GradScaler()
    # eval a epoch
    #los=validate(args, model, data_loader_val, device)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            args, model, data_loader_train, optimizer, device, epoch,scaler, args.clip_max_norm,
        )
        lr_scheduler.step()

        val_stats = validate(args, model, data_loader_val, device)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'validation_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 10 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            if val_stats['accu'] > best_accu:
                checkpoint_paths.append(output_dir / 'best_checkpoint.pth')
                best_accu=val_stats['accu']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'val_accu': val_stats['accu']
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransVG training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
# import torchinfo
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from timm.utils import accuracy, AverageMeter, ModelEma
import timm

# assert timm.__version__ == "0.5.4"  # version check
from timm.models.layers import trunc_normal_
import timm.optim.optim_factory as optim_factory
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay_spikformer as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models
from engine_finetune import train_one_epoch, evaluate
from timm.data import create_loader
from qp_quant_ptq import (
    attach_layer_names,
    set_global_mpq_options,
    PTQCalibConfig,
    ptq_calibrate_model,
    set_ptq_enabled,
    strip_quantizer_state_dict,
    load_state_dict_ignore_mismatch,
    set_full_precision_mode,
    set_ptq_mode,
    print_quant_state,
)


# def change(train_set):
#     import torchvision.datasets as datasets
#     # 只加载训练集
#     # train_set = datasets.ImageNet(root='/path/to/imagenet', split='train', download=True)
#     # 取前500类
#     subset_classes = list(range(500))
#     train_set.targets = [c for c in train_set.targets if c in subset_classes]
#     # 每类取前400张
#     filtered_indices = []
#     for c in subset_classes:
#         cls_indices = [i for i, t in enumerate(train_set.targets) if t == c]
#         filtered_indices.extend(cls_indices[:400])

#     train_set.samples = [train_set.samples[i] for i in filtered_indices]
#     train_set.targets = [train_set.targets[i] for i in filtered_indices]
#     return train_set

def get_args_parser():
    # important params
    parser = argparse.ArgumentParser(
        "MAE fine-tuning for image classification", add_help=False
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=200, type=int)  # 20/30(T=4)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument(
        "--data_path", default="/data/dataset", type=str, help="dataset path"
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="spikformer_8_384_CAFormer",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--model_mode",
        default="ms",
        type=str,
        help="Mode of model to train",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",  # 1e-5,2e-5(T=4)
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=1.0,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR"
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup", type=float, default=0, help="mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--cutmix", type=float, default=0, help="cutmix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # * Finetuning params

    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )
    parser.add_argument("--time_steps", default=1, type=int)
    parser.add_argument("--eval_time_steps", default=4, type=int)

    # Dataset parameters

    parser.add_argument(
        "--nb_classes",
        default=10,
        type=int,
        help="number of the classification types",
    )

    parser.add_argument('--data_set', default='IMNET', type=str,
                    choices=['CIFAR10', 'CIFAR100', 'IMNET'],
                    help='dataset type')

    parser.add_argument(
        "--output_dir",
        default="/raid/ligq/htx/spikemae/output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir",
        default="/raid/ligq/htx/spikemae/output_dir",
        help="path where to tensorboard log",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--MODEL_EMA", default=True, type=int)
    parser.add_argument("--MODEL_EMA_DECAY", default=0.99996, type=float)
    parser.add_argument("--resume", default=None, help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    #PTQ mode参数
    parser.add_argument('--ptq_enable', action='store_true')
    parser.add_argument('--ptq_calibrate', action='store_true')
    parser.add_argument('--ptq_target_avg_bit', type=float, default=4.0)
    parser.add_argument('--ptq_fisher_batches', type=int, default=32)
    parser.add_argument('--ptq_candidate_bits', type=int, nargs='+', default=[2, 3, 4, 6, 8])
    parser.add_argument('--ptq_rescaw_gamma_mode', type=str, default='layer_l1_mean',
                        choices=['layer_l1_mean', 'channel_l1_mean', 'layer_maxabs', 'channel_maxabs'])
    parser.add_argument('--ptq_rescaw_clip_value', type=float, default=1.0)
    parser.add_argument('--ptq_grid_multipliers', type=float, nargs='+', default=[1.0])
    parser.add_argument('--ptq_save_mapping', type=str, default='')
    parser.add_argument('--ptq_drop_wq_state_on_load', action='store_true', default=True)
    parser.add_argument("--ptq_use_logit_sensitivity", action="store_true")
    parser.add_argument("--ptq_sensitivity_batches", default=8, type=int)
    parser.add_argument("--ptq_sensitivity_temperature", default=2.0, type=float)

    #剪枝参数
    parser.add_argument("--auto_prune_tgsrs", action="store_true")
    parser.add_argument("--tgsrs_rank_batches", default=6, type=int)
    parser.add_argument("--tgsrs_hook_group_size", default=8, type=int)
    parser.add_argument("--tgsrs_rank_dir", default="./rank_conv_tgsrs", type=str)
    parser.add_argument("--tgsrs_compress_rate", default="[0.30]*999", type=str)
    parser.add_argument("--tgsrs_min_keep", default=1, type=int)
    parser.add_argument("--tgsrs_skip_layer_ids", default="", type=str)
    parser.add_argument("--tgsrs_bit_protect_omega", default=0.30, type=float)

    return parser

def main(args):
    misc.init_distributed_mode(args)
#     args.model = "spikformer_8_512_CAFormer"
    # args.resume = './output_dir/checkpoint-32.pth'

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    print("nb_classes =", args.nb_classes)
#     dataset_train = change(dataset_train)
#     print('==========================')
#     print('dataset',dataset_train)
#     print('==========================')
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )
    if args.ptq_enable:
        set_global_mpq_options(
            candidate_bits=tuple(args.ptq_candidate_bits),
            rescaw_gamma_mode=str(args.ptq_rescaw_gamma_mode),
            rescaw_clip_value=float(args.ptq_rescaw_clip_value),
            grid_multipliers=tuple(args.ptq_grid_multipliers),
        )
    else:
        # 注意：这里不是启用 4bit，只是初始化量化层配置。
        # 真正是否量化由 PTQWeightQuantizer.enabled 控制。
        set_global_mpq_options(
            candidate_bits=(2, 4, 6, 8),
            gamma_percentile=0.999,
            hard_gumbel=False,
        )
    model = models.__dict__[args.model](nb_classes=args.nb_classes)
    model.T = args.time_steps
    attach_layer_names(model)

    # 默认先关闭所有 PTQ。
    # 后面如果 args.ptq_enable and args.ptq_calibrate，会重新校准并开启。
    if not args.ptq_enable:
        set_full_precision_mode(
            model,
            clear_masks=(not args.auto_prune_tgsrs),
            verbose=True,
        )
    model_ema = None
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location="cpu")
        print("Load pre-trained checkpoint from: %s" % args.finetune)

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint_model = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint_model = checkpoint["state_dict"]
        else:
            checkpoint_model = checkpoint

        # 去掉 DDP 保存时可能带的 module. 前缀
        checkpoint_model = {
            k.replace("module.", "", 1): v
            for k, v in checkpoint_model.items()
        }

        for k in ["head.weight", "head.bias", "head.fc.weight", "head.fc.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape[0] != args.nb_classes:
                # print(f"Remove mismatched key from checkpoint: {k}")
                del checkpoint_model[k]

        if args.ptq_enable:
            checkpoint_model = strip_quantizer_state_dict(checkpoint_model, drop_all_wq_state=True)
            msg = load_state_dict_ignore_mismatch(model, checkpoint_model)
        else:
            msg = model.load_state_dict(checkpoint_model, strict=False)
        # print(msg)

        if not args.ptq_enable:
            set_full_precision_mode(
                model,
                clear_masks=(not args.auto_prune_tgsrs),
                verbose=True,
            )
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    # ---------------- PTQ calibration before DDP/optimizer ----------------
    # It is better to run PTQ before DistributedDataParallel wrapping; otherwise each rank may
    # calibrate on a different data shard and produce slightly different bit mappings.
    if args.ptq_enable and args.ptq_calibrate:
        ptq_map_path = args.ptq_save_mapping
        if ptq_map_path == '' and args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            ptq_map_path = os.path.join(args.output_dir, 'ptq_bit_mapping.pth')


        ptq_cfg = PTQCalibConfig(
            candidate_bits=tuple(args.ptq_candidate_bits),
            target_avg_bit=float(args.ptq_target_avg_bit),
            fisher_batches=int(args.ptq_fisher_batches),
            rescaw_gamma_mode=str(args.ptq_rescaw_gamma_mode),
            rescaw_clip_value=float(args.ptq_rescaw_clip_value),
            grid_multipliers=tuple(args.ptq_grid_multipliers),
            save_path=ptq_map_path,
        )
        ptq_cfg.use_logit_sensitivity = bool(args.ptq_use_logit_sensitivity)
        ptq_cfg.sensitivity_batches = int(args.ptq_sensitivity_batches)
        ptq_cfg.sensitivity_temperature = float(args.ptq_sensitivity_temperature)

        ptq_report = ptq_calibrate_model(
            model=model_without_ddp,
            dataloader=data_loader_train,
            criterion=torch.nn.CrossEntropyLoss(),
            device=device,
            cfg=ptq_cfg,
        )

        print(f"[PTQ] actual average bit = {ptq_report['avg_bit']:.4f}")

        if misc.is_main_process():
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'model': model_without_ddp.state_dict(),
                'ptq_report': ptq_report,
                'args': args,
            }, os.path.join(args.output_dir, 'checkpoint-ptq.pth'))
            print('[PTQ] saved checkpoint-ptq.pth')

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        args.weight_decay,
        # no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
        model_mode = args.model_mode
    )
    #     exit(0)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr) # lamb
    # optimizer = optim_factory.Lamb(param_groups, trust_clip=True, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )
    if not args.ptq_enable:
        set_full_precision_mode(
            model_without_ddp,
            clear_masks=(not args.auto_prune_tgsrs),
            verbose=True,
        )
        # ---------------- tg-prune (run once BEFORE finetune) ----------------
    if args.auto_prune_tgsrs and (not args.eval):
        from tg_srs_prune import auto_prune_tgsrs, TGSRSConfig

        skip_ids = []
        if isinstance(args.tgsrs_skip_layer_ids, str) and len(args.tgsrs_skip_layer_ids.strip()) > 0:
            skip_ids = [
                int(x.strip())
                for x in args.tgsrs_skip_layer_ids.split(",")
                if len(x.strip()) > 0
            ]

        cfg_prune = TGSRSConfig(
            limit_batches=args.tgsrs_rank_batches,
            device=str(device),
            hook_group_size=args.tgsrs_hook_group_size,
            save_score_dir=args.tgsrs_rank_dir,
            compress_rate=args.tgsrs_compress_rate,
            min_keep=args.tgsrs_min_keep,
            skip_layer_ids_1based=skip_ids,
            bit_protect_omega=args.tgsrs_bit_protect_omega,
        )

        masks = auto_prune_tgsrs(
            model_without_ddp,
            data_loader_train,
            criterion=torch.nn.CrossEntropyLoss(),
            cfg=cfg_prune,
        )

        print(f"[TG-SRS] done. masked layers = {len(masks)}")

        if misc.is_main_process():
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, "tgsrs_pruned_before_finetune.pth")

            torch.save(
                {
                    "model": model_without_ddp.state_dict(),
                    "masks": {k: v.cpu() for k, v in masks.items()},
                    "auto_prune_tgsrs": True,
                    "tgsrs_cfg": cfg_prune.__dict__,
                },
                save_path,
            )

            print(f"[TG-SRS] saved pruned model to: {save_path}")

        model.to(device)
        model.train()
    # ---------------------------------------------------------------------------
    if args.MODEL_EMA:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but
        # before SyncBN and DDP wrapper
        print(f"Using EMA...")
        model_ema = ModelEma(
            model,
            decay=args.MODEL_EMA_DECAY,
        )
    if args.eval:
            # eval: T=4
        model_without_ddp.T = args.eval_time_steps
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    print_quant_state(model_without_ddp, max_lines=12)
    start_time = time.time()
    max_accuracy = 0.0
    save_interval=100
    best_acc = 0
    best_epoch = 0
    best_state=None
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        # train: T=1
        model_without_ddp.T = args.time_steps
        train_stats,model_ema = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            mixup_fn,
            log_writer=log_writer,
            args=args,
            model_ema=model_ema)
        if args.output_dir and ((epoch + 1) % save_interval == 0 or epoch == args.epochs):
            print("Saving model at epoch:", epoch)
            misc.save_model(
                args=args,
                model=model_ema,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
        # eval: T=4
        model_without_ddp.T = args.eval_time_steps
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")
        # if args.output_dir and test_stats["acc1"] > best_acc:
        #     print("Saving model at epoch:", epoch)
        #     misc.save_model(
        #         args=args,
        #         model=model,
        #         model_without_ddp=model_without_ddp,
        #         optimizer=optimizer,
        #         loss_scaler=loss_scaler,
        #         epoch=epoch,
        #     )
        if args.output_dir and test_stats["acc1"] > best_acc:
            best_acc = float(test_stats["acc1"])
            best_epoch = epoch

            # 保存最佳权重到内存（建议存 model_without_ddp，避免DDP包装问题）
            best_state = {k: v.cpu().clone() for k, v in model_without_ddp.state_dict().items()}
            print(f"[Best] Update: acc1={best_acc:.3f}% at epoch {best_epoch}")


        if log_writer is not None:
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_acc5", test_stats["acc5"], epoch)
            log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)
        
        def compute_bit_param(model, w_bits=4):
            total = 0
            for name, p in model.named_parameters():
                if "weight" in name:
                    total += p.numel() * w_bits
                else:
                    total += p.numel() * 32
            return total / 32 / 1e6  # 转成 M
        bit_param = compute_bit_param(model, w_bits=4)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
            # "param":bit_param,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")
   
    # ---------------- save best checkpoint at the end ----------------
    if args.output_dir and misc.is_main_process() and best_state is not None:
        best_path = os.path.join(args.output_dir, "checkpoint-best.pth")
        payload = {
            "model": best_state,
            "best_acc1": best_acc,
            "best_epoch": best_epoch,
            "args": args,
        }
        torch.save(payload, best_path)
        print(f"Saved BEST checkpoint to: {best_path} (acc1={best_acc:.3f}%, epoch={best_epoch})")
    # -----------------------------------------------------------------

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

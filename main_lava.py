# This code is built on top of DINO github repo and modified by Islam Nassar (islam.nassar@monash.edu):
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torchvision.transforms import InterpolationMode
from sentence_transformers import SentenceTransformer


import utils
import vision_transformer as vits
from vision_transformer import LAVAHead
from eval_or_predict import eval_or_predict
from eval_or_predict_knn import eval_or_predict_knn
from losses import MyHingeLoss

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('LAVA', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'deit_tiny', 'deit_small'] + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--pretrained_weights', default=None, type=str, help="Path to pretrained weights if needed")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--load_backbone_only', default=False, type=utils.bool_flag,
                        help="""If set, only the backbone pretrained weights will be loaded and the head weights will
                        be ignored. We set this to true for a semi-sup run and to False for a LAVA finetuning run""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the LAVA head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=False, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the LAVA head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # LAVA parameters
    parser.add_argument('--word_vec_path', default='/home/inas0003/data/extra/numberbatch-en-19.08_128D.dict.pkl',
                        type=str, help="""Path for class labels emebdding pkl dictionary""")
    parser.add_argument('--use_sentence_transformer', type=utils.bool_flag, default=True, help="""If set, sentence 
    transformer language model will be used to obtain sentence_embeddings, otherwise conceptnet embdgs will be used.""")
    parser.add_argument('--transformer_language_model', default='paraphrase-mpnet-base-v2',
                        type=str, help="""The language model to be used for obtaining semantic embeddings, 
                        e.g. 'clip-ViT-B-32' for a full list, see https://www.sbert.net/docs/pretrained_models.html""")
    parser.add_argument('--class_prefix', default=None, type=str,
                        help="""An optional prefix (prompt) to be added before class labels before obtaining embeddings. 
                        e.g. 'a photo of a' """)
    parser.add_argument('--start_pl_epoch', default=-1, type=int, help="""epoch at which pseudo-loss will start to be 
    considered and dino loss will be switched off""")
    parser.add_argument('--freeze_backbone_till', default=-1, type=int, help="""backbone will be frozen till these expire""")
    parser.add_argument('--freeze_student_backbone', default=False, type=utils.bool_flag,
                        help="if set, student backbone will be frozen. This is to only be used for head finetuning runs")
    parser.add_argument('--student_cls_temp', default=0.1, type=float, help="""student classifier head temperature, i.e.,  
    for supervised and pseudo-labelling loss""")
    parser.add_argument('--lam_dino', default=1.0, type=float, help="""Modulation factor for dino loss""")
    parser.add_argument('--lam_sup', default=1.0, type=float, help="""Modulation factor for supervised loss""")
    parser.add_argument('--lam_pl', default=1.0, type=float, help="""Modulation factor for pseudo-label loss""")
    parser.add_argument('--lam_sem', default=9.0, type=float, help="""Modulation factor for semantic loss""")
    parser.add_argument('--lr_scaler', default=1.0, type=float, help="""Learning rate will be reduced by this factor
    once pseudo-label loss starts""")
    parser.add_argument('--fusion_tie_breaker', default=2, type=int, help="""to decide about which head to use in case 
    a draw while fusing prediction. 1 denotes one-hot, 2 denotes semantic head, 3 denotes KNN """)
    parser.add_argument('--restore_epoch_from_chkpt', default=True, type=utils.bool_flag,
                        help="Whether restore epoch from a checkpoint if found, useful to set to False if you are continuing"
                             "training DINO (without supervision not pseudo-label). ")
    parser.add_argument('--project_dino_space_to_class_space', default=False, type=utils.bool_flag,
                        help="Whether to connect dino space output directly to the class space or to have a split from last layer input.  ")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct unlabelled images loaded on one GPU.')
    parser.add_argument('--mu', default=1, type=int,
                        help='Ratio of unlabelled to labelled data per batch. '
                             'For example, if batch_size_per_gpu is 64, '
                             'the effective images loaded on one gpu per batch would be 64 + 64/mu ')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--break_epoch', default=-1, type=int, help='Overrides for epoch to allow using same schedules')
    parser.add_argument('--freeze_last_layer', default=0, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--eval', type=utils.bool_flag, default=True, help="""If set, evaluation on 'val' directory data 
    will be performed at the beginning of every epoch of training.""")
    parser.add_argument('--eval_linear', type=utils.bool_flag, default=True, help="""If set, linear eval will take place + KNN""")
    parser.add_argument('--eval_freq', type=int, default=2, help="""Frequency at which evaluation using val set will be performed""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str,
        help='Please specify path to the training data directory which which contains train, val and test directories.'
             'train directory should contain labelled and unlabelled subdirectories')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--write_predictions_to_disk', type=utils.bool_flag, default=False, help="""If set, 
    predictions of oh and sem will be written to disk in the final epodh""")
    parser.add_argument('--saveckp_freq', default=200, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--saveresumeckp_freq', default=1, type=int, help='Save resume checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--dist_master_port", default="29500", type=str, help="""MASTER_PORT used for torch distributed.
    Change that if you want to run two scripts simultaneously on the same local machine""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train_lava(args):
    if not utils.is_dist_avail_and_initialized():
        utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    conf_print = "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    print(conf_print)
    if utils.is_main_process():
        (Path(args.output_dir)/'config.txt').open('w').write(conf_print)
    cudnn.benchmark = True

    # ============ preparing data ... ============
    tic = time.time()
    transform = DataAugmentationLAVA(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        add_unlab_weak_aug_crop=False,
    )
    # create unlabeled training dataset
    dataset_unlabelled = datasets.ImageFolder(Path(args.data_path)/'train', transform=transform)
    sampler_unlabelled = torch.utils.data.DistributedSampler(dataset_unlabelled, shuffle=True)
    data_loader_unlabelled = torch.utils.data.DataLoader(
        dataset_unlabelled,
        sampler=sampler_unlabelled,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    dataset_labelled = datasets.ImageFolder(Path(args.data_path)/'train/labelled', transform=transform.global_transfo3)
    sampler_labelled = torch.utils.data.DistributedSampler(dataset_labelled, shuffle=True)
    data_loader_labelled = torch.utils.data.DataLoader(
        dataset_labelled,
        sampler=sampler_labelled,
        batch_size=int(args.batch_size_per_gpu / args.mu),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    num_classes = len(dataset_labelled.classes)
    print(f"Data loaded in {int((time.time()-tic)/60)} minutes: there are {len(dataset_unlabelled)} images, "
          f"out of which {len(dataset_labelled)} are labelled.")
    # ============ building label semantic embedding matrix ... ============
    classes = dataset_labelled.classes
    # load imagenet mapping dictionary
    imgnt_cls_lkup = pickle.load((Path(__file__).parent.resolve() / 'aux/imagenet_class_lookup_dict.pkl').open('rb'))
    # handle imagenet class names if needed
    classes = [imgnt_cls_lkup[cls] if cls in imgnt_cls_lkup else cls for cls in classes]
    print(f'classes: {classes}')
    if args.use_sentence_transformer:
        model = SentenceTransformer(args.transformer_language_model)
        if args.class_prefix is not None:
            classes = [f'{args.class_prefix.strip()} {cls}' for cls in classes]
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        emb_matrix = torch.tensor(model.encode(classes)).cuda()
    else:
        class_2_emb_dict = utils.get_labels2wv_dict(classes, args.word_vec_path)
        emb_matrix = torch.tensor([class_2_emb_dict[cls] for cls in classes]).cuda()
    print(f'Initialised Label Embedding Matrix with shape: {emb_matrix.shape}')
    sem_emb_dim = emb_matrix.shape[1]
    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=0.1,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknown architecture: {args.arch}")

    # load pretrained weights of backbone if specified
    if args.pretrained_weights is not None and args.load_backbone_only:
        print('**************************Loading Backbone weights without Head**************************')
        utils.load_pretrained_weights(student, args.pretrained_weights, args.checkpoint_key, args.arch,
                                      args.patch_size, remove_backbone_prefix=True, filter_if_error=True)

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, LAVAHead(
        embed_dim,
        args.out_dim,
        num_classes,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
        emb_dim=sem_emb_dim,
        project_dino_space=args.project_dino_space_to_class_space
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        LAVAHead(embed_dim,
                      args.out_dim,
                      num_classes,
                      args.use_bn_in_head,
                      emb_dim=sem_emb_dim,
                      project_dino_space=args.project_dino_space_to_class_space),
    )

    # load pretrained weights pf backbone + head if specified
    if args.pretrained_weights is not None and not args.load_backbone_only:
        print('**************************Loading Backbone weights including Head**************************')
        utils.load_pretrained_weights(student, args.pretrained_weights, args.checkpoint_key, args.arch,
                                      args.patch_size, remove_backbone_prefix=False, filter_if_error=True)

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    # freeze student backbone if needed (will only be used for head finetuning runs)
    if args.freeze_student_backbone:
        print('********************\nFreezing Student Backbone\n*********************')
        for n, p in student.named_parameters():
            if 'backbone' in n:
                p.requires_grad = False
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    lava_loss = LAVALoss(
        args.out_dim,
        # total number of crops = 3 global crops + local_crops_number + 1 labelled  - one more global crop compared to dino
        args.local_crops_number + 3,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        num_classes=num_classes,
        student_classifier_temp=args.student_cls_temp,
        emb_matrix=emb_matrix,
        lambda_dino=args.lam_dino,
        lambda_sup=args.lam_sup,
        lambda_pl=args.lam_pl,
        lambda_sem=args.lam_sem,

    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader_unlabelled),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader_unlabelled),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader_unlabelled))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        lava_loss=lava_loss,
    )
    if args.restore_epoch_from_chkpt:
        start_epoch = to_restore["epoch"]
        print(f'**********Setting epoch to {start_epoch}***************')
    else:
        start_epoch = 0
        print(f'***********Checkpoint model has been trained for {to_restore["epoch"]} epochs,'
              f' but resetting epoch counter to zero ')

    start_time = time.time()
    print("Starting LAVA training !")
    end_epoch = args.epochs if args.break_epoch < 0 else args.break_epoch
    for epoch in range(start_epoch, end_epoch):
        eval_stats = {}
        top1_knn = [0]
        preds_oh_sem = None
        preds_knn = None
        top1_fused = 0
        if args.eval and utils.is_main_process() and (not epoch % args.eval_freq or epoch == end_epoch - 1):
            if args.eval_linear:
                print('**************Evaluating Linearly**************')
                preds_oh_sem , target, eval_stats = eval_or_predict(teacher, args, eval=True, emb_matrix=emb_matrix)
                preds_oh, preds_sem = preds_oh_sem
                if args.write_predictions_to_disk:
                    (Path(args.output_dir)/'preds_oh.txt').open('a').write('\n'.join(preds_oh) + "\n")
                    (Path(args.output_dir) / 'preds_sem.txt').open('a').write('\n'.join(preds_sem) + "\n")
                    (Path(args.output_dir) / 'true_labels.txt').open('a').write('\n'.join(target) + "\n")
            print('**************Evaluating KNN**************')
            top1_knn, top5_knn, preds_knn = eval_or_predict_knn(args, [10, 20, 30, 40], eval=True) # [3, 5, 10, 20]
            if preds_oh_sem is not None and preds_knn is not None:
                preds_knn = utils.fuse_predictions(*preds_knn[:3])  # first we fuse the first three KNN prediction lists
                preds_fused = utils.fuse_predictions(preds_oh, preds_sem, preds_knn, tie_breaker=args.fusion_tie_breaker) # then we fuse oh, sem and knn
                top1_fused = np.mean(np.array(preds_fused) == np.array(target)) * 100
        dist.barrier()  # so that other dist processes wait till evaluation is complete

        data_loader_unlabelled.sampler.set_epoch(epoch)
        data_loader_labelled.sampler.set_epoch(epoch)
        if epoch == start_epoch:
            global dl_lab
            dl_lab = iter(data_loader_labelled)
        # ============ Adjusting loss from clustering space to class space... ============
        if epoch < args.start_pl_epoch:
            lava_loss.lambda_pl = torch.tensor(0).cuda()
        elif epoch == args.start_pl_epoch:
            print(f'Changing lava loss from generic prediction space to class prediction space')
            lava_loss.lambda_pl = torch.tensor(args.lam_pl).cuda()
            lava_loss.lambda_dino = torch.tensor(0).cuda()
        # ============ training one epoch of LAVA ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, lava_loss,
            data_loader_unlabelled, data_loader_labelled, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'lava_loss': lava_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        if (args.saveresumeckp_freq and epoch % args.saveresumeckp_freq == 0) or epoch == end_epoch - 1:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in eval_stats.items()},
                     'epoch': epoch}
        if np.mean(top1_knn) > 0:
            log_stats['val_KNN_top1'] = np.mean(top1_knn[:2])
        if top1_fused > 0:
            log_stats['top1_fused'] = top1_fused
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, lava_loss, data_loader_unlabelled,
                    data_loader_labelled, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                    epoch, fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}-{}]'.format(epoch, args.epochs, args.break_epoch if args.break_epoch > 0 else args.epochs)
    global dl_lab
    for it, (images, dummy) in enumerate(metric_logger.log_every(data_loader_unlabelled, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader_unlabelled) * epoch + it  # global training iteration
        lr_scaler = 1. if epoch < args.start_pl_epoch else args.lr_scaler
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it] / lr_scaler
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
        # load labeled images
        try:
            images_labelled, lbs = next(dl_lab)
        except StopIteration:
            dl_lab = iter(data_loader_labelled)
            images_labelled, lbs = next(dl_lab)
        images.append(images_labelled)
        # obtain split map to know how to split after forward
        # (could have been calculated via batch_size and mu but kept here for further flexibility, e.g. dynamic batch size)
        split_map = [batch.shape[0] for batch in images]
        # move tensors to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        lbs = lbs.cuda()
        # teacher and student forward passes + compute lava loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_lava_output, teacher_cls_output, teacher_sem_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_lava_output, student_cls_output, student_sem_output = student(images)
            loss, loss_stats = lava_loss(student_lava_output, teacher_lava_output, student_cls_output,
                                               teacher_cls_output, student_sem_output, teacher_sem_output,
                                               lbs, split_map, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            utils.cancel_gradients_backbone(epoch, student,
                                              args.freeze_backbone_till)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            utils.cancel_gradients_backbone(epoch, student,
                                            args.freeze_backbone_till)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(**loss_stats)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LAVALoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, num_classes, emb_matrix,
                 student_temp=0.1, center_momentum=0.9, student_classifier_temp=0.1,
                 lambda_dino=1., lambda_sup=1., lambda_pl=1., lambda_sem=1.):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center_class", torch.zeros(1, num_classes))
        self.register_buffer("lambda_dino", torch.tensor(lambda_dino))
        self.register_buffer("lambda_sup", torch.tensor(lambda_sup))
        self.register_buffer("lambda_pl", torch.tensor(lambda_pl))
        self.register_buffer("lambda_sem", torch.tensor(lambda_sem))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.student_classifier_temp = student_classifier_temp
        self.emb_matrix = emb_matrix

    def forward(self, student_lava_output, teacher_lava_output, student_cls_output, teacher_cls_output,
                student_sem_output, teacher_sem_output, labels, split_map, epoch):
        """
        lava losses comprising dino loss, supervised loss and pseudo-label losses.
        """
        def _calculate_loss(student_output, teacher_output, soft_classes):
            student_out = student_output / self.student_temp
            student_out = torch.split(student_out, split_map)
            student_out = student_out[:-1]  # remove the labeled image student prediction


            # teacher centering and sharpening
            temp = self.teacher_temp_schedule[epoch]
            center = self.center if soft_classes else self.center_class
            teacher_out = F.softmax((teacher_output - center) / temp, dim=-1)
            teacher_out = teacher_out.detach().chunk(2)

            loss = 0
            n_loss_terms = 0
            for iq, q in enumerate(teacher_out):
                for v in range(len(student_out)):
                    if v == iq:
                        # we skip cases where student and teacher operate on the same view
                        continue
                    loss_batch = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                    loss += loss_batch.mean()
                    n_loss_terms += 1
            loss /= n_loss_terms
            self.update_center(teacher_output, soft_classes)
            return loss

        lava_loss = _calculate_loss(student_lava_output, teacher_lava_output, soft_classes=True) * self.lambda_dino
        pl_loss = _calculate_loss(student_cls_output, teacher_cls_output, soft_classes=False) * self.lambda_pl

        # supervised loss
        student_cls_out = torch.split(student_cls_output, split_map)
        criterion = lambda inp, targ: F.cross_entropy(inp, targ, reduction='none')
        sup_loss = self.lambda_sup * criterion(student_cls_out[-1] / self.student_temp, labels).mean() #student_cls_out[-1]: classifier logits for labelled weakly aug
        # semantic loss
        student_sem_out = torch.split(student_sem_output, split_map)
        # criterion = lambda inp, targ: F.mse_loss(inp, targ, reduction='none')
        # criterion = lambda inp, targ: 1 - F.cosine_similarity(inp, targ)
        criterion = MyHingeLoss(margin=0.4, dimension=self.emb_matrix.shape[1])
        sem_loss = self.lambda_sem * criterion(student_sem_out[-1], self.emb_matrix[labels])#.mean()
        # combine all losses
        total_loss = lava_loss + sup_loss + pl_loss + sem_loss
        with torch.no_grad():
            loss_stats = {'loss_lava': lava_loss.item(), 'loss_sup': sup_loss.item(),
                          'loss_pl': pl_loss.item(), 'loss_sem': sem_loss.item()}
            loss_stats = {k: v for k, v in loss_stats.items() if v > 1e-6}
        return total_loss, loss_stats

    @torch.no_grad()
    def update_center(self, teacher_output, soft_classes):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        if soft_classes:
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        else:
            self.center_class = self.center_class * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationLAVA(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, add_unlab_weak_aug_crop=False):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # third global crop - with minimal transform for LAVA loss
        self.global_transfo3 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 1.), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])
        self.add_unlab_weak_aug_crop = add_unlab_weak_aug_crop

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        if self.add_unlab_weak_aug_crop:
            crops.append(self.global_transfo3(image))
        return crops

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LAVA', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_lava(args)



# edits for imgnt dict
# imgnt_cls_lkup['n02012849'] = 'crane_the_bird'
# imgnt_cls_lkup['n02963159'] = 'cardigan_the_clothing_item'
# imgnt_cls_lkup['n02113186'] = 'Cardigan_Welsh_corgi'
# imgnt_cls_lkup['n03126707'] = 'crane_the_construction_tool'
# imgnt_cls_lkup['n03710637'] = 'maillot_the_bathing_suit'
# imgnt_cls_lkup['n03710721'] = 'maillot_the_tank_suit'

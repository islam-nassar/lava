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
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision.transforms import InterpolationMode

try:
    import utils
    import vision_transformer as vits
except:
    import modelrun.solution.utils as utils
    import modelrun.solution.vision_transformer as vits


def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = ReturnIndexDataset(os.path.join(args.data_path, "train/labelled"), transform=transform)
    dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "val" if args.eval else 'test'), transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val, args.use_cuda)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        imgs_tr = [str(Path(elem[0])) for elem in dataset_train.imgs]
        imgs_val = [str(Path(elem[0])) for elem in dataset_val.imgs]
        torch.save(imgs_tr, os.path.join(args.dump_features, "trainimgs.pth"))
        torch.save(imgs_val, os.path.join(args.dump_features, "testimgs.pth"))
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    classes = dataset_train.classes
    imgs = [Path(elem[0]).name for elem in dataset_val.imgs]
    return train_features, test_features, train_labels, test_labels, classes, imgs


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, classes):
    num_classes = len(classes)
    k = min(k, train_features.shape[0])
    top1, top5, total = 0.0, 0.0, 0
    probs_list = []
    preds_list = []
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(min(k, similarity.shape[1]), largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        if args.eval:
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, 5).sum().item()
            total += targets.size(0)
        probs_list.append(probs)
        preds_list.append(predictions[:, 0])
    top1 = (top1 * 100.0 / total) if total else 0.
    top5 = (top5 * 100.0 / total) if total else 0.
    probability_vec = nn.functional.softmax(torch.cat(probs_list) * T, dim=-1).cpu().numpy()
    prediction_vec = list(torch.cat(preds_list).cpu().numpy())
    prediction_vec = list(np.array(classes)[prediction_vec])
    return top1, top5, probability_vec, prediction_vec



class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx

def parse_knn_args():
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
                        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--k', default=20, type=int,
                        help='K for single use. Use nb_knn above if more than one value but need to edit code')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
                        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--eval', default=False, type=utils.bool_flag,
                        help="Eval or test")
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
                        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
            already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
            distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--dist_master_port", default="29500", type=str, help="""master port""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--output_dir', default=None, type=str)
    args = parser.parse_args()

    return args

def main(args):
    try:
        utils.init_distributed_mode(args)
    except:
        dist.new_group(backend='nccl',
                       ranks=list(range(utils.get_world_size()))
                       )
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.load_features:
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
    else:
        # need to extract features !
        train_features, test_features, train_labels, test_labels, classes, imgs = extract_feature_pipeline(args)

    if utils.get_rank() == 0:
        if args.use_cuda:
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()

        print("Features are ready!\nStart the k-NN classification.")
        # top1_list = []
        # top5_list = []
        # for k in args.nb_knn:
        top1, top5, probs, preds = knn_classifier(train_features, train_labels,
            test_features, test_labels, args.k, args.temperature, classes)
        # top1_list.append(top1)
        # top5_list.append(top5)
        print(f"{args.k}-NN classifier result: Top1: {top1}, Top5: {top5}")
        deterministic = pd.DataFrame((zip(imgs, preds)), columns=['id', 'class'])
        probabilistic = pd.DataFrame(probs, columns=classes)
        probabilistic.insert(0, 'id', imgs)
        # write dataframes to output_dir
        deterministic.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
        probabilistic.to_csv(os.path.join(args.output_dir, 'prob_predictions.csv'), index=False)
    dist.barrier()

if __name__ == '__main__':

    args = parse_knn_args()
    main(args)


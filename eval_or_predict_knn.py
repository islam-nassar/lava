
import os
import argparse
import numpy as np
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

def eval_or_predict_knn(args, k_list, temperature=0.07, eval=True, use_cuda=True):
    cudnn.benchmark = True
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # this is to support fewshot experiments when repeating the dataset multiple times
    tr_path = os.path.join(args.data_path, "train_no_rep/labelled")
    if not os.path.exists(tr_path):
        tr_path = os.path.join(args.data_path, "train/labelled")
    dataset_train = ReturnIndexDataset(tr_path, transform=transform)
    dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "val" if eval else "test"), transform=transform)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size_per_gpu * 16,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu * 16,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val/test imgs.")

    # ============ preparing model ... ============
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    model.cuda()
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        pretrained_weights = os.path.join(args.output_dir, 'checkpoint.pth')
    elif args.pretrained_weights is not None:
        if os.path.exists(args.pretrained_weights):
            pretrained_weights = args.pretrained_weights
        else:
            print('Pretrained weights nonexistent, cancelling KNN')
            return [-1], [-1], [-1]
    else:
        print('No pretrained weights found, cancelling KNN an returning')
        return [-1], [-1], [-1]
    utils.load_pretrained_weights(model, pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()
    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train)
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val)


    train_features = nn.functional.normalize(train_features, dim=1, p=2)
    test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()


    if use_cuda:
        train_features = train_features.cuda()
        test_features = test_features.cuda()
        train_labels = train_labels.cuda()
        test_labels = test_labels.cuda()

    print("Features are ready!\nStart the k-NN classification.")
    num_classes = len(dataset_train.classes)
    top1_list, top5_list, preds_list = [], [], []
    for k in k_list:
        top1, top5, preds = knn_classifier(train_features, train_labels,
            test_features, test_labels, k, temperature, num_classes)
        print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
        top1_list.append(top1)
        top5_list.append(top5)
        preds_list.append(preds)
    classes = dataset_train.classes
    prediction_vec_list = [list(np.array(classes)[sub_list]) for sub_list in preds_list]
    return top1_list, top5_list, prediction_vec_list

@torch.no_grad()
def extract_features(model, data_loader, use_cuda=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 20):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        feats = model(samples).clone()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        if use_cuda:
            features.index_copy_(0, index, feats)
        else:
            features.index_copy_(0, index.cpu(), feats.cpu())
    return features

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes):
    k = min(k, train_features.shape[0])
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], min(test_labels.shape[0], 100)
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
    all_preds = []
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
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
        all_preds.extend(predictions[:, 0].cpu().tolist())  # we are interested in the top1 prediction only
        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        if num_classes >=5:
            top5 = top5 + correct.narrow(1, 0, 5).sum().item()
        else:
            top5 = 0
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5, all_preds


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx

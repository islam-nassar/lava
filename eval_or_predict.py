
import os
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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

def eval_or_predict(model, args, eval=True, emb_matrix=None):
    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val" if eval else "test"),
                                       transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu * 4,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_val)} val/test imgs.")

    # ============ building network ... ============
    model.cuda()
    model.eval()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} is in use.")

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    all_preds = []
    all_preds_sem = []
    correct = []
    with torch.no_grad():
        for inp, target in metric_logger.log_every(val_loader, 50, header) if eval else val_loader:
            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            _, output_cls, output_cls_sem = model(inp)
            _, preds = torch.max(output_cls, dim=1)
            # transform semantic outputs to logits by obtaining pairwise sim with emb_matrix
            output_cls_sem = F.cosine_similarity(output_cls_sem.unsqueeze(1), emb_matrix.unsqueeze(0), dim=-1)
            _, preds_sem = torch.max(output_cls_sem, dim=1)

            all_preds.append(preds)
            all_preds_sem.append(preds_sem)
            correct.append(target)

            num_labels = output_cls.shape[-1]
            if eval:
                acc1 = utils.accuracy(output_cls, target)[0]
                acc1_sem = utils.accuracy(output_cls_sem, target)[0]
                if num_labels >= 5:
                    acc5 = utils.accuracy(output_cls, target, topk=(5,))[0]
                    acc5_sem = utils.accuracy(output_cls_sem, target, topk=(5,))[0]

                loss = nn.CrossEntropyLoss()(output_cls, target)

            batch_size = inp.shape[0]
            if eval:
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc1_sem'].update(acc1_sem.item(), n=batch_size)
                if num_labels >= 5:
                    metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
                    metric_logger.meters['acc5_sem'].update(acc5_sem.item(), n=batch_size)
        if eval:
            if num_labels >= 5:
                print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                  .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
                print('* Semantic Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
                      .format(top1=metric_logger.acc1_sem, top5=metric_logger.acc5_sem))
            else:
                print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
                  .format(top1=metric_logger.acc1, losses=metric_logger.loss))
                print('* Semantic Acc@1 {top1.global_avg:.3f}'
                      .format(top1=metric_logger.acc1_sem))

        predictions = torch.cat(all_preds)
        predictions_sem = torch.cat(all_preds_sem)
        correct = list(torch.cat(correct).cpu().numpy())
        model.train()
        prediction_vec = list(predictions.cpu().numpy())
        prediction_vec_sem = list(predictions_sem.cpu().numpy())
        classes = dataset_val.classes
        prediction_vec = list(np.array(classes)[prediction_vec])
        prediction_vec_sem = list(np.array(classes)[prediction_vec_sem])
        correct = list(np.array(classes)[correct])
        return (prediction_vec, prediction_vec_sem), correct, \
               {k: meter.global_avg for k, meter in metric_logger.meters.items()}
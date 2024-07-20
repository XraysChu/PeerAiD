from torchvision.datasets import CocoDetection
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import argparse
import math
import os
import sys
from typing import Iterable

import torch

import utils.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

from attack_function import *
import json
import time
import datetime

# Define the path to the COCO dataset
data_path = "../data/kitti/"

# Define the transform to apply to the images
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add more transformations if needed
])

# Create an instance of the CocoDetection dataset
# train_dataset = CocoDetection(root=data_path + "train2017", annFile=data_path + "annotations/instances_train2017.json", transform=transform)
# val_dataset = CocoDetection(root=data_path + "val2017", annFile=data_path + "annotations/instances_val2017.json", transform=transform)
# dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

# # Use a data loader to load the dataset
# # data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
# batch_size = 4
# # Create data loaders.
# train_dataset = DataLoader(train_dataset, batch_size=batch_size, collate_fn=utils.collate_fn)
# val_dataset = DataLoader(val_dataset, batch_size=batch_size, collate_fn=utils.collate_fn)

# data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)

# for i in range(10):
#     sample = dataset[i]
#     img, target = sample
#     print(f"{type(img) = }\n{type(target) = }\n{type(target[0]) = }\n{target[0].keys() = }\n{target[0]['bbox'] = }")
# num_classes = train_dataset.coco.cats
# print(f"Number of classes: {len(num_classes)}")
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
    
def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks
class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

import datasets.transforms as T
def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

from pathlib import Path

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    # if args.dataset_file == 'coco':
    return build(image_set, args)

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--temperature', type = float, default=6, help='temperature of the distillation term in the loss of the student model.')
    parser.add_argument('--re_kd_temperature', type = float, default=1, help='the temperature parameter of the distillation term in the loss of the peer model.')
    parser.add_argument('--lamb1', type = float, default=1, help='lambda1 hyperparameter in the loss of the student model.')
    parser.add_argument('--lamb2', type = float, default=1, help='lambda2 hyperparameter in the loss of the student model.')
    parser.add_argument('--lamb3', type = float, default=1, help='lambda3 hyperparameter in the loss of the student model.')
    parser.add_argument('--gamma1', type = float, default=1, help='gamma1 hyperparameter in the loss of the peer model.')
    parser.add_argument('--gamma2', type = float, default=1, help='gamma2 hyperparameter in the loss of the peer model.')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
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

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # print(f"samples: {samples}, \nattributes: {dir(samples)}")
        # print(f"targets: {targets}, \nattributes: {dir(targets)}")
        # print(samples.shape)
        # tensors, masks = samples.decompose()
        # print(f"tensors: {tensors.shape}")
        # print(f"masks: {masks.shape}")

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def adv_train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    epsilon = 8/255 
    alpha = 2/255 
    k_train = 10
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        outputs = model(samples)
        
        adversary_student_training = randomAttack(model, epsilon, alpha)

        adv = adversary_student_training.perturb(samples, targets, k_train)
        adv_outputs = model(adv)
        # CElosses = nn.CrossEntropyLoss(adv_outputs, targets)


        # loss_dict = criterion(outputs, targets)
        loss_dict = criterion(adv_outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_two_peers(model: torch.nn.Module, criterion: torch.nn.Module,
                              model_p: torch.nn.Module, criterion_p: torch.nn.Module,
                              data_loader: Iterable, optimizer: torch.optim.Optimizer,
                              peer_optimizer: torch.optim.Optimizer, device: torch.device, 
                              epoch: int, args):
    # student train 
    model.train()
    criterion.train()
    # peer train
    model_p.train()
    criterion_p.train()


    T = args.temperature

    peer_train_loss = 0
    student_train_loss = 0
    correct = 0
    total = 0
    correct_student = 0
    total_student = 0
    epsilon = 8/255 
    alpha = 2/255 

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for batch_idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        peer_optimizer.zero_grad()

        model_p.eval()
        with torch.no_grad():
            peer_logits = model_p(samples)

        model_p.train() 

        adversary_student_training = randomAttack(model, epsilon, alpha)

        adv = adversary_student_training.perturb(samples, targets, 10)

        model.train()
        # adv.detach_()

        adv_outputs = model_p(adv)
        pred_logits_peer = adv_outputs['pred_logits']
        # print(type(adv_outputs))
        # print(adv_outputs)
        # adv_outputs_x, _ = adv_outputs.decompose()
        student_logit_target = model(adv)
        pred_logits_stu = student_logit_target['pred_logits']
        # student_logit_target_x, _ = student_logit_target.decompose()

        loss_peer = nn.KLDivLoss()(F.log_softmax(pred_logits_peer/args.re_kd_temperature, dim=1), F.softmax(pred_logits_stu/args.re_kd_temperature, dim=1)) * args.gamma2 * args.re_kd_temperature * args.re_kd_temperature 

        # print(type(loss_peer))
        # print(loss_peer)
        # peer_train_loss += loss_peer.item()
        # _, predicted = adv_outputs.max(1)

        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()

        # peer_outputs = adv_outputs.clone()
        # peer_outputs.detach_()

        # adv.detach_()
        # student_outputs = model(adv)

        # _, predicted_student = student_outputs.max(1)
        # total_student += targets.size(0)
        # correct_student += predicted_student.eq(targets).sum().item()

        # samples.detach_()
        logit_nat = model(samples)
        pred_logits_nat = logit_nat['pred_logits']
        model.train()

        loss_student = nn.KLDivLoss()(F.log_softmax(pred_logits_stu / T, dim=1),
                                      F.softmax(pred_logits_peer / T, dim=1)) * args.lamb2 * T * T + \
                       nn.KLDivLoss()(F.log_softmax(pred_logits_stu / T, dim=1),
                                      F.softmax(pred_logits_nat / T, dim=1)) * args.lamb3 * T * T 
                                    #   + criterion * args.lamb1 

        loss_total = loss_peer + loss_student
        loss_total.backward()

        peer_optimizer.step()
        optimizer.step()

        student_train_loss += loss_student.item()

        if batch_idx % 20 == 0:
            print('\nCurrent batch:', str(batch_idx))

        # if batch_idx == 20:
        #     print('Current adversarial peer train accuracy:', str(predicted.eq(targets).sum().item() * 100 / targets.size(0)))
        #     print('Current adversarial peer train loss:', loss_peer.item())
        #     print('Current adversarial student train accuracy:', str(predicted_student.eq(targets).sum().item() * 100 / targets.size(0)))
        #     print('Current adversarial student train loss:', loss_student.item())
        loss_dict = criterion(adv_outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        metric_logger.update(loss=loss_total.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

    # Synchronize and print the averaged metrics
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # Print total adversarial peer and student training statistics
    # print('\nTotal adversarial peer train accuracy:', 100. * correct / total)
    # print('Total adversarial peer train loss:', peer_train_loss)
    # print('Total adversarial student train loss:', student_train_loss)
    # print('Total adversarial student train accuracy:', 100. * correct_student / total_student)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# train the model   
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# train_dataset = CocoDetection(data_path, ann_file= , transforms=make_coco_transforms('train'), return_masks=False)
# val_dataset = CocoDetection(data_path, '../data/kitti/annotations/', transforms=make_coco_transforms('val'), return_masks=False)
from models import detr, build_model
from models.detr import DETR
parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

dataset_train = build_dataset(image_set='train', args=args)
sampler_train = torch.utils.data.RandomSampler(dataset_train)
batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size=4, drop_last=True)
data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=1)

dataset_val = build_dataset(image_set='val', args=args)
sampler_train = torch.utils.data.RandomSampler(dataset_val)
batch_sampler_val = torch.utils.data.BatchSampler(
        sampler_train, batch_size=4, drop_last=True)
data_loader_val = DataLoader(dataset_val, batch_sampler=batch_sampler_val,
                                   collate_fn=utils.collate_fn, num_workers=1)


model, criterion, postprocessors = build_model(args)

model.to(device)
# Load the DETR model
# model = DETR(num_classes=3,backbone="resnet50",transformer="transformer",num_queries=50,aux_loss=True)
# model = model.to(device)
# print(model)
# model.train(train_dataloader, val_dataloader, epochs=10)
model_without_ddp = model
param_dicts = [
    {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": args.lr_backbone,
    },
]

optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
# def train(dataloader, model, loos_fn, optimizer):
#     model.train()
#     for i, (images, targets) in enumerate(dataloader):
#         images = images.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#         output = model(images)
#         loss = loss_fn(output, targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if i % 10 == 0:
#             print(f"Iteration {i}, loss = {loss}")
# train_one_epoch_two_peers(model, criterion, model, criterion, data_loader_train, optimizer, optimizer, device, 1, args)
# train_one_epoch(model, criterion, data_loader_train, optimizer, device, 1)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
output_dir = Path(args.output_dir)
start_time = time.time()
for epoch in range(args.start_epoch, args.epochs):
    # if args.distributed:
    #     sampler_train.set_epoch(epoch)
    train_stats = train_one_epoch(
        model, criterion, data_loader_train, optimizer, device, epoch,
        args.clip_max_norm)
    lr_scheduler.step()
    if args.output_dir:
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        # extra checkpoint before LR drop and every 100 epochs
        if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

    # test_stats, coco_evaluator = evaluate(
    #     model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
    # )

    # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
    #                 **{f'test_{k}': v for k, v in test_stats.items()},
    #                 'epoch': epoch,
    #                 'n_parameters': n_parameters}

    # if args.output_dir and utils.is_main_process():
    #     with (output_dir / "log.txt").open("a") as f:
    #         f.write(json.dumps(log_stats) + "\n")

    #     # for evaluation logs
    #     if coco_evaluator is not None:
    #         (output_dir / 'eval').mkdir(exist_ok=True)
    #         if "bbox" in coco_evaluator.coco_eval:
    #             filenames = ['latest.pth']
    #             if epoch % 50 == 0:
    #                 filenames.append(f'{epoch:03}.pth')
    #             for name in filenames:
    #                 torch.save(coco_evaluator.coco_eval["bbox"].eval,
    #                             output_dir / "eval" / name)

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))
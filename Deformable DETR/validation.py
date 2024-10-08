import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate_without_criterion
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='ablation_outputs_all10_resnet101/checkpoint0089.pth', help='resume from checkpoint')
    parser.add_argument('--output_dir', default='mapout_all10_resnet101', help='path where to save, empty for no saving')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='/mnt/disk1/data0/jxt/dataset/data/allsides_10/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)

    # Model parameters
    parser.add_argument('--backbone', default='resnet101', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # Transformer parameters
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int, help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Learning parameters (even though they are not used in evaluation, we define them to avoid errors)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    return parser

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, _, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model

    dataset_val = build_dataset(image_set='val', args=args)
    cat_ids = dataset_val.coco.getCatIds()
    print("Category IDs in dataset:", cat_ids)
    if args.distributed:
        sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    base_ds = get_coco_api_from_dataset(dataset_val)

    print("Start evaluation")
    start_time = time.time()
    test_stats, coco_evaluator = evaluate_without_criterion(model, postprocessors, data_loader_val, base_ds, device, args.output_dir)
    if args.output_dir:
        utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, Path(args.output_dir) / "eval.pth")

    # Print mAP for each category and all categories
    if coco_evaluator is not None and 'bbox' in coco_evaluator.coco_eval:
        coco_eval = coco_evaluator.coco_eval['bbox']
        coco_eval.accumulate()
        coco_eval.summarize()

        aps = coco_eval.eval['precision']
        ars = coco_eval.eval['recall']
        if aps.shape[2] > 0:
            cat_ids = coco_eval.params.catIds
            max_cat_id = aps.shape[2]
            valid_cat_ids = [cat_id for cat_id in cat_ids if cat_id <= max_cat_id]

            iou_index = coco_eval.params.iouThrs.tolist().index(0.5)
            # Calculate average precision, recall and F1 for each category
            ap_per_category = {cat_id: np.mean(aps[:, :, cat_id - 1, 0, -1]) for cat_id in valid_cat_ids}
            precision_per_category = {cat_id: np.mean(aps[iou_index, :, cat_id - 1, 0, -1]) for cat_id in valid_cat_ids}
            recall_per_category = {cat_id: np.mean(ars[iou_index, cat_id - 1, 0, -1]) for cat_id in valid_cat_ids}
            f1_per_category = {cat_id: 2 * (precision_per_category[cat_id] * recall_per_category[cat_id]) /
                                       (precision_per_category[cat_id] + recall_per_category[cat_id])
            if (precision_per_category[cat_id] + recall_per_category[cat_id]) > 0 else 0
                               for cat_id in valid_cat_ids}

            print("mAP (IoU=0.5) for each category:")
            for cat_id in sorted(ap_per_category.keys()):  # 确保按类别ID排序
                mAP = ap_per_category[cat_id]
                print(f"{cat_id:02d} : {mAP:.6f}")

            print("Precision (IoU=0.5) for each category:")
            for cat_id in sorted(precision_per_category.keys()):  # 确保按类别ID排序
                precision = precision_per_category[cat_id]
                print(f"{cat_id:02d} : {precision:.6f}")

            print("Recall (IoU=0.5) for each category:")
            for cat_id in sorted(recall_per_category.keys()):  # 确保按类别ID排序
                recall = recall_per_category[cat_id]
                print(f"{cat_id:02d} : {recall:.6f}")

            print("F1 Score (IoU=0.5) for each category:")
            for cat_id in sorted(f1_per_category.keys()):  # 确保按类别ID排序
                f1 = f1_per_category[cat_id]
                print(f"{cat_id:02d} : {f1:.6f}")

            # Extract overall metrics
            overall_mAP = np.mean(list(ap_per_category.values()))  # 手动计算的平均mAP
            overall_precision = np.mean(list(precision_per_category.values()))  # 手动计算的平均precision
            overall_recall = np.mean(list(recall_per_category.values()))  # 手动计算的平均Recall
            overall_f1 = np.mean(list(f1_per_category.values()))  # 手动计算的平均F1
            print(f"\nCalculated overall mAP: {overall_mAP:.6f}")
            print(f"Calculated overall precision (IoU=0.5): {overall_precision:.6f}")
            print(f"Calculated overall recall (IoU=0.5): {overall_recall:.6f}")
            print(f"Calculated overall F1 (IoU=0.5): {overall_f1:.6f}")
            # COCO evaluation provided overall mAP
            coco_overall_mAP = coco_eval.stats[1]  # AP @ IoU=0.5
            print(f"\nCOCO provided overall mAP (IoU=0.5): {coco_overall_mAP:.6f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
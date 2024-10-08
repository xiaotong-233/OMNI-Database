import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import nms
import torchvision.ops as ops

def faster_rcnn():
    backbone = torchvision.models.resnet50(pretrained=True)
    for param in backbone.parameters():
        param.requires_grad = True
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    backbone.out_channels = 2048
    anchor_generator = AnchorGenerator(sizes=((8, 16, 32),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    return model

def filter_valid_boxes(boxes):
    valid_indices = torch.where((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))[0]
    return boxes[valid_indices]

def process_image(image, model, targets, training=False, detection_threshold=0.5, nms_threshold=0.3):
    # 确保图像和模型在同一个设备上
    device = next(model.parameters()).device
    image = image.to(device).float()
    boxes = filter_valid_boxes(targets['boxes'])
    num_boxes = boxes.shape[0]
    labels = torch.ones((num_boxes,), dtype=torch.int64, device=image.device)
    targets = {'boxes': boxes, 'labels': labels}
    if training:
        # 在训练模式下，只计算损失值
        loss_dict = model([image], [targets])
        return loss_dict, None, None
    else:
        # 在评估模式下，获取检测结果和全局特征
        model.eval()  # 切换到评估模式
        with torch.no_grad():
            outputs = model([image])
        processed_output = []
        cropped_images_list = []  # 存储裁剪的图像
        for output in outputs:
            if isinstance(output, dict) and 'scores' in output:
                scores = output['scores']
                boxes = output['boxes']
                labels = output.get('labels', torch.ones(scores.shape, dtype=torch.int64, device=scores.device))
                indices = torch.where(scores > detection_threshold)[0]
                if indices.numel() > 0:
                    scores = scores[indices]
                    boxes = boxes[indices]
                    labels = labels[indices]
                    keep = nms(boxes, scores, nms_threshold)
                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    processed_output.append({'boxes': boxes, 'scores': scores, 'labels': labels})

                    # 裁剪和保存图像
                    for box in boxes:
                        # 调整boxes格式为 [batch_idx, x1, y1, x2, y2]
                        box = torch.cat((torch.tensor([0], device=device), box)).unsqueeze(0)
                        cropped_image = ops.roi_align(image.unsqueeze(0), box, output_size=(224, 224))
                        cropped_images_list.append(cropped_image)
        model.train()  # 切换回训练模式
        return None, processed_output, cropped_images_list

def fasterrcnn(model_path):
    backbone = faster_rcnn()
    if model_path is not None:
        # 加载预训练模型
        state_dict = torch.load(model_path)
        if 'state_dict' in state_dict:
            backbone.load_state_dict(state_dict['state_dict'])
        else:
            backbone.load_state_dict(state_dict)
    backbone = backbone.to('cuda')
    def model_process(batch_images, batch_targets, training=True):
        results = []
        loss_list = []
        cropped_images_list = []  # 用于存储每个批次的裁剪图像
        for image, targets in zip(batch_images, batch_targets):
            if training:
                loss_dict, _, _ = process_image(image, backbone, targets, training=True)
                loss_list.append(loss_dict)
            else:
                _, processed_output, cropped_images = process_image(image, backbone, targets, training=False)
                cropped_images_list.extend(cropped_images)
                if processed_output:
                    results.append(processed_output[0])
                else:
                    result_dict = {'boxes': torch.empty((0, 4), device='cuda'), 'scores': torch.empty((0,), device='cuda')}
                    results.append(result_dict)
        if training:
            return loss_list
        else:
            return results, cropped_images_list
    return backbone, model_process


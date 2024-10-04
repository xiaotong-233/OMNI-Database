import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign, nms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class CustomRoIHeads(RoIHeads):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_features = None  # 初始化一个属性来存储特征

        # 添加一个box_head来处理特征
        self.box_head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),  # 将特征图的空间维度降低到1x1
            torch.nn.Flatten(),  # 展平特征
            torch.nn.Linear(256, 2048)  # 将通道维度映射到2048
        )
        num_classes = 2
        self.box_predictor = FastRCNNPredictor(2048, num_classes)

    def forward(self, features, proposals, image_shapes, targets=None):
        # 在调用父类的forward方法之前，清除之前存储的特征
        self.box_features = []
        # 调用父类的forward方法
        result, losses = super().forward(features, proposals, image_shapes, targets)
        # 保存RoIAlign后的特征
        if not self.training:
            self.box_features = [self.box_head(feature) for feature in features.values()]
        return result, losses

class CustomFasterRCNN(FasterRCNN):
    def __init__(self, num_classes):
        backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        backbone.out_channels = 256  # FPN输出的通道数
        anchor_generator = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
        roi_heads = CustomRoIHeads(
            box_roi_pool=roi_pooler,
            box_head=torch.nn.Sequential(),  # 使用一个空的Sequential作为box_head
            box_predictor=torch.nn.Linear(2048, num_classes),  # 随意设置一个box_predictor
            fg_iou_thresh=0.5, bg_iou_thresh=0.5,
            batch_size_per_image=128, positive_fraction=0.25,
            bbox_reg_weights=None, score_thresh=0.05,
            nms_thresh=0.5, detections_per_img=100
        )
        super().__init__(backbone, num_classes, rpn_anchor_generator=anchor_generator)
        # 设置自定义的RoIHeads
        self.roi_heads = roi_heads
        # 添加一个额外的全连接层来处理全局特征的维度变化
        self.global_feature_head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),  # 将特征图大小变为 1x1
            torch.nn.Flatten(start_dim=1),  # 展平特征图
            torch.nn.Linear(256, 2048)  # 将通道数从 256 变为 2048
        )

    def forward(self, images, targets=None):
        # 调用父类的forward方法获取模型输出
        outputs = super().forward(images, targets)
        if self.training:
            return outputs  # 在训练模式下返回损失
        # 在评估模式下提取全局特征和局部特征
        global_features = []
        local_features = []
        if not isinstance(images, torch.Tensor):
            images = torch.stack([img for img in images])
        features = self.backbone(images)
        highest_level_features = features['pool']
        for feature_map in highest_level_features:
            # 确保特征图有批次维度
            if feature_map.dim() == 3:
                feature_map = feature_map.unsqueeze(0)
            global_feat = self.global_feature_head(feature_map)
            global_features.append(global_feat)  # 移除批次维度
        if hasattr(self.roi_heads, 'box_features'):
            for box_feature in self.roi_heads.box_features:
                local_feat = torch.flatten(box_feature, 1)
                local_features.append(local_feat)
        return outputs, global_features, local_features

def filter_valid_boxes(boxes):
    valid_indices = torch.where((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))[0]
    return boxes[valid_indices]

def process_image(image, model, targets, training=False, detection_threshold=0.5, nms_threshold=0.4):
    boxes = filter_valid_boxes(targets['boxes'])
    num_boxes = boxes.shape[0]
    labels = torch.ones((num_boxes,), dtype=torch.int64, device=image.device)
    targets = {'boxes': boxes, 'labels': labels}
    if training:
        # 在训练模式下，只计算损失值
        loss_dict = model([image], [targets])
        return loss_dict, None, None
    else:
        # 在评估模式下，获取检测结果和特征
        model.eval()  # 切换到评估模式
        with torch.no_grad():
            outputs, global_features, local_features = model([image])
        processed_output = []
        for output in outputs:
            if 'scores' in output:
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
        model.train()  # 切换回训练模式
        return None, processed_output, global_features, local_features

def fasterrcnn():
    model = CustomFasterRCNN(num_classes=2)
    model = model.to('cuda')

    def model_process(batch_images, batch_targets, training=True):
        results = []
        global_features_list = []
        local_features_list = []
        loss_list = []
        for image, targets in zip(batch_images, batch_targets):
            image, targets['boxes'] = image.to('cuda'), targets['boxes'].to('cuda')
            if training:
                loss_dict, _, _ = process_image(image, model, targets, training=True)
                loss_list.append(loss_dict)
            else:
                _, processed_output, global_features, local_features = process_image(image, model, targets, training=False)
                if processed_output:
                    results.append(processed_output[0])
                else:
                    results.append(torch.empty((0, 4), device='cuda'))
                global_features_list.append(global_features[0])
                local_features_list.append(local_features[0])
        if training:
            return loss_list
        else:
            return results, global_features_list, local_features_list
    return model, model_process

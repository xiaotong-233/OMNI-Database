import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from .swin_transformer import swin_transformer_tiny, swin_transformer_small, swin_transformer_base
from .resnet import resnet18, resnet34, resnet50, resnet101
from .graph import create_e_matrix
from .graph_edge_model import GEM
from .basic_block import *


# Gated GCN Used to Learn Multi-dimensional Edge Features and Node Features
class GNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # GNN Matrix: E x N
        # Start Matrix Item:  define the source node of one edge
        # End Matrix Item:  define the target node of one edge
        # Algorithm details in Residual Gated Graph Convnets: arXiv preprint arXiv:1711.07553
        # or Benchmarking Graph Neural Networks: arXiv preprint arXiv:2003.00982v3


        dim_in = self.in_channels
        dim_out = self.in_channels

        self.U1 = nn.Linear(dim_in, dim_out, bias=False)
        self.V1 = nn.Linear(dim_in, dim_out, bias=False)
        self.A1 = nn.Linear(dim_in, dim_out, bias=False)
        self.B1 = nn.Linear(dim_in, dim_out, bias=False)
        self.E1 = nn.Linear(dim_in, dim_out, bias=False)

        self.U2 = nn.Linear(dim_in, dim_out, bias=False)
        self.V2 = nn.Linear(dim_in, dim_out, bias=False)
        self.A2 = nn.Linear(dim_in, dim_out, bias=False)
        self.B2 = nn.Linear(dim_in, dim_out, bias=False)
        self.E2 = nn.Linear(dim_in, dim_out, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)

        self.act = nn.ReLU()

        self.init_weights_linear(dim_in, 1)

    def init_weights_linear(self, dim_in, gain):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        self.U1.weight.data.normal_(0, scale)
        self.V1.weight.data.normal_(0, scale)
        self.A1.weight.data.normal_(0, scale)
        self.B1.weight.data.normal_(0, scale)
        self.E1.weight.data.normal_(0, scale)

        self.U2.weight.data.normal_(0, scale)
        self.V2.weight.data.normal_(0, scale)
        self.A2.weight.data.normal_(0, scale)
        self.B2.weight.data.normal_(0, scale)
        self.E2.weight.data.normal_(0, scale)


    def forward(self, x, edge):
        x = x.unsqueeze(0)
        device = x.device
        num_anchors = x.size(1)
        start, end = create_e_matrix(num_anchors)
        start_node = Variable(start, requires_grad=False).to(device)
        end_node = Variable(end, requires_grad=False).to(device)
        bnv1 = nn.BatchNorm1d(num_anchors).to(device)
        bne1 = nn.BatchNorm1d(num_anchors * num_anchors).to(device)

        bnv2 = nn.BatchNorm1d(num_anchors).to(device)
        bne2 = nn.BatchNorm1d(num_anchors * num_anchors).to(device)

        bn_init(bnv1)
        bn_init(bne1)
        bn_init(bnv2)
        bn_init(bne2)


        # GNN Layer 1:
        res = x
        Vix = self.A1(x)  # V x d_out
        Vjx = self.B1(x)  # V x d_out
        # edge = edge.squeeze(0)
        e = self.E1(edge)  # E x d_out
        edge = edge + self.act(bne1(torch.einsum('ev, bvc -> bec', (end_node, Vix)) + torch.einsum('ev, bvc -> bec',(start_node, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, num_anchors, num_anchors, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V1(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start_node, Ujx))  # E x H_out
        Uix = self.U1(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end_node.t(), e * Ujx)) / num_anchors  # V x H_out
        x = self.act(res + bnv1(x))
        res = x

        # GNN Layer 2:
        Vix = self.A2(x)  # V x d_out
        Vjx = self.B2(x)  # V x d_out
        e = self.E2(edge)  # E x d_out
        edge = edge + self.act(bne2(torch.einsum('ev, bvc -> bec', (end_node, Vix)) + torch.einsum('ev, bvc -> bec', (start_node, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, num_anchors, num_anchors, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V2(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start_node, Ujx))  # E x H_out
        Uix = self.U2(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end_node.t(), e * Ujx)) / num_anchors  # V x H_out
        x = self.act(res + bnv2(x))
        return x, edge


class Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Head, self).__init__()
        # The head of network
        # Input: the feature maps x from backbone
        # Output: the AU recognition probabilities cl And the logits cl_edge of edge features for classification
        # Modules: 1. AFG extracts individual Au feature maps U_1 ---- U_N
        #          2. GEM: graph edge modeling for learning multi-dimensional edge features
        #          3. Gated-GCN for graph learning with node and multi-dimensional edge features
        # sc: individually calculate cosine similarity between node features and a trainable vector.
        # edge fc: for edge prediction

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.linear_block = LinearBlock(self.in_channels, self.in_channels)
        self.edge_extractor = GEM(self.in_channels, self.num_classes)
        self.gnn = GNN(self.in_channels, self.num_classes)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.relu = nn.ReLU()
        nn.init.xavier_uniform_(self.sc)

    def forward(self, initial_features, cropped_features, image_indices):
        # Initialize list to store outputs for each image's graph
        output_features = []

        # Get unique image indices to process each image's samples
        for i in range(initial_features.shape[0]):  # Loop over each image in the batch
            # Select features corresponding to current image
            mask = (image_indices == i)
            image_cropped_features = cropped_features[mask]
            node_features = image_cropped_features.mean(dim=-2)

            # Process image-specific features
            # You can adjust the GEM and GNN calls based on actual signature and functionality
            edge_features = self.edge_extractor(image_cropped_features, initial_features[i])
            edge_features = edge_features.mean(dim=-2)
            node_features, edge_features = self.gnn(node_features, edge_features)
            # Example of processing node features, e.g., computing classification logits
            node_features = F.normalize(node_features, p=2, dim=-1)
            sc = self.relu(self.sc)
            sc_normalized = F.normalize(sc, p=2, dim=-1)
            cl = torch.matmul(node_features, sc_normalized.t())
            cl = cl.squeeze(0)
            output_features.append(cl)

        # Concatenate or otherwise combine all the results
        return output_features
    # def forward(self, initial_features, cropped_features):
    #     # AFG
    #     # device = initial_features.device
    #     # num_anchors = cropped_features.size(0)
    #     # class_linear_layers = nn.ModuleList(
    #     #     [LinearBlock(self.in_channels, self.in_channels).to(device) for _ in range(num_anchors)])
    #     # f_u = [layer(cropped_features).unsqueeze(1) for layer in class_linear_layers]
    #     # f_u = torch.cat(f_u, dim=1)  # （13，13，512）
    #     f_v = cropped_features.mean(dim=-2)  # （13，512）
    #
    #     # MEFL
    #     f_e = self.edge_extractor(cropped_features, initial_features)
    #     f_e = f_e.mean(dim=-2)
    #     f_v, f_e = self.gnn(f_v, f_e)
    #
    #     b, n, c = f_v.shape
    #     sc = self.sc
    #     sc = self.relu(sc)
    #     sc = F.normalize(sc, p=2, dim=-1)
    #     cl = F.normalize(f_v, p=2, dim=-1)
    #     cl = (cl * sc.view(1, n, c)).sum(dim=-1, keepdim=False)
    #     return cl


class MEFARG(nn.Module):
    def __init__(self, num_classes=10, backbone1='resnet101', backbone2='resnet152'):
        super(MEFARG, self).__init__()
        self.backbone1 = self.create_backbone(backbone1)
        self.backbone2 = self.create_backbone(backbone2)
        # self.in_channels = self.backbone1.num_features if 'transformer' in backbone1 else 2048
        # self.out_channels = self.in_channels // 2 if 'transformer' in backbone1 else self.in_channels // 4
        #
        # self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        # Assuming backbone1 is transformer and backbone2 is ResNet
        self.in_channels_backbone1 = self.backbone1.num_features if 'transformer' in backbone1 else 2048
        self.in_channels_backbone2 = self.backbone2.num_features if 'transformer' in backbone2 else 2048  # Using 512 if ResNet18 or ResNet34

        self.out_channels_backbone1 = self.in_channels_backbone1 // 2 if 'transformer' in backbone1 else self.in_channels_backbone1 // 4
        self.out_channels_backbone2 = self.in_channels_backbone2 // 2 if 'transformer' in backbone2 else self.in_channels_backbone2 // 4

        self.global_linear1 = LinearBlock(self.in_channels_backbone1, self.out_channels_backbone1)
        self.global_linear2 = LinearBlock(self.in_channels_backbone2, self.out_channels_backbone2)
        self.head = Head(self.out_channels_backbone1, num_classes)

    def create_backbone(self, backbone_name):
        if 'transformer' in backbone_name:
            if backbone_name == 'swin_transformer_tiny':
                backbone = swin_transformer_tiny()
            elif backbone_name == 'swin_transformer_small':
                backbone = swin_transformer_small()
            else:
                backbone = swin_transformer_base()
            backbone.head = None  # Remove the original classification head
        elif 'resnet' in backbone_name:
            if backbone_name == 'resnet18':
                backbone = resnet18()
            elif backbone_name == 'resnet34':
                backbone = resnet34()
            elif backbone_name == 'resnet101':
                backbone = resnet101()
            else:
                backbone = resnet50()
            backbone.fc = None  # Remove the fully connected layer
        else:
            raise Exception("Error: wrong backbone name: ", backbone_name)
        return backbone

    def forward(self, batch_images, cropped_images):
        initial_features = self.backbone2(batch_images)  # （4，256，512）
        initial_features = self.global_linear2(initial_features)

        all_cropped_images = []
        image_indices = []
        for img_idx, crops in enumerate(cropped_images):
            merged_crops = torch.cat(crops, dim=0)
            all_cropped_images.append(merged_crops)
            image_indices.extend([img_idx] * merged_crops.shape[0])

        all_cropped_images = torch.cat(all_cropped_images, dim=0)
        image_indices = torch.tensor(image_indices)

        cropped_features = self.backbone1(all_cropped_images)
        cropped_features = self.global_linear1(cropped_features)
        cl = self.head(initial_features, cropped_features, image_indices)
        return cl

# # 移除边特征
# import torch
# import torch.nn as nn
# import numpy as np
# import torch.nn.functional as F
# from torch.autograd import Variable
# import math
# from .swin_transformer import swin_transformer_tiny, swin_transformer_small, swin_transformer_base
# from .resnet import resnet18, resnet34, resnet50, resnet101
# from .graph import create_e_matrix
# from .basic_block import *
#
# # Gated GCN Used to Learn Node Features (Edge features removed)
# class GNN(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(GNN, self).__init__()
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#
#         dim_in = self.in_channels
#         dim_out = self.in_channels
#
#         # Define linear layers for node features (removing edge-related layers)
#         self.U1 = nn.Linear(dim_in, dim_out, bias=False)
#         self.V1 = nn.Linear(dim_in, dim_out, bias=False)
#         self.A1 = nn.Linear(dim_in, dim_out, bias=False)
#         self.B1 = nn.Linear(dim_in, dim_out, bias=False)
#
#         self.U2 = nn.Linear(dim_in, dim_out, bias=False)
#         self.V2 = nn.Linear(dim_in, dim_out, bias=False)
#         self.A2 = nn.Linear(dim_in, dim_out, bias=False)
#         self.B2 = nn.Linear(dim_in, dim_out, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax(2)
#         self.act = nn.ReLU()
#
#         self.init_weights_linear(dim_in, 1)
#
#     def init_weights_linear(self, dim_in, gain):
#         scale = gain * np.sqrt(2.0 / dim_in)
#         self.U1.weight.data.normal_(0, scale)
#         self.V1.weight.data.normal_(0, scale)
#         self.A1.weight.data.normal_(0, scale)
#         self.B1.weight.data.normal_(0, scale)
#
#         self.U2.weight.data.normal_(0, scale)
#         self.V2.weight.data.normal_(0, scale)
#         self.A2.weight.data.normal_(0, scale)
#         self.B2.weight.data.normal_(0, scale)
#
#     def forward(self, x, edge=None):  # Removed edge-related operations
#         x = x.unsqueeze(0)
#         device = x.device
#         num_anchors = x.size(1)
#         start, end = create_e_matrix(num_anchors)
#         start_node = Variable(start, requires_grad=False).to(device)
#         end_node = Variable(end, requires_grad=False).to(device)
#         bnv1 = nn.BatchNorm1d(num_anchors).to(device)
#         bnv2 = nn.BatchNorm1d(num_anchors).to(device)
#
#         bn_init(bnv1)
#         bn_init(bnv2)
#
#         # GNN Layer 1 (node feature processing only)
#         res = x
#         Vix = self.A1(x)  # V x d_out
#         Vjx = self.B1(x)  # V x d_out
#         Ujx = self.V1(x)  # V x H_out
#         Uix = self.U1(x)  # V x H_out
#         x = Uix + torch.einsum('ve, bvc -> bvc', (start_node.t(), Ujx)) / num_anchors  # Node feature update
#         x = self.act(res + bnv1(x))
#         res = x
#
#         # GNN Layer 2 (node feature processing only)
#         Vix = self.A2(x)  # V x d_out
#         Vjx = self.B2(x)  # V x d_out
#         Ujx = self.V2(x)  # V x H_out
#         Uix = self.U2(x)  # V x H_out
#         x = Uix + torch.einsum('ve, bvc -> bvc', (start_node.t(), Ujx)) / num_anchors
#         x = self.act(res + bnv2(x))
#
#         return x, None  # Edge is now set to None since edge features are removed
#
# class Head(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(Head, self).__init__()
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         self.linear_block = LinearBlock(self.in_channels, self.in_channels)
#         self.gnn = GNN(self.in_channels, self.num_classes)
#         self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
#         self.relu = nn.ReLU()
#         nn.init.xavier_uniform_(self.sc)
#
#     def forward(self, initial_features, cropped_features, image_indices):
#         # Initialize list to store outputs for each image's graph
#         output_features = []
#
#         # Get unique image indices to process each image's samples
#         for i in range(initial_features.shape[0]):  # Loop over each image in the batch
#             # Select features corresponding to current image
#             mask = (image_indices == i)
#             image_cropped_features = cropped_features[mask]
#             node_features = image_cropped_features.mean(dim=-2)
#
#             # Pass node features through GNN (no edge features)
#             node_features, _ = self.gnn(node_features, None)
#
#             # Compute classification logits using node features
#             node_features = F.normalize(node_features, p=2, dim=-1)
#             sc = self.relu(self.sc)
#             sc_normalized = F.normalize(sc, p=2, dim=-1)
#             cl = torch.matmul(node_features, sc_normalized.t())
#             cl = cl.squeeze(0)
#             output_features.append(cl)
#
#         # Concatenate or otherwise combine all the results
#         return output_features
#
# class MEFARG(nn.Module):
#     def __init__(self, num_classes=10, backbone1='resnet50', backbone2='resnet101'):
#         super(MEFARG, self).__init__()
#         self.backbone1 = self.create_backbone(backbone1)
#         self.backbone2 = self.create_backbone(backbone2)
#
#         self.in_channels_backbone1 = self.backbone1.num_features if 'transformer' in backbone1 else 2048
#         self.in_channels_backbone2 = self.backbone2.num_features if 'transformer' in backbone2 else 2048
#
#         self.out_channels_backbone1 = self.in_channels_backbone1 // 2 if 'transformer' in backbone1 else self.in_channels_backbone1 // 4
#         self.out_channels_backbone2 = self.in_channels_backbone2 // 2 if 'transformer' in backbone2 else self.in_channels_backbone2 // 4
#
#         self.global_linear1 = LinearBlock(self.in_channels_backbone1, self.out_channels_backbone1)
#         self.global_linear2 = LinearBlock(self.in_channels_backbone2, self.out_channels_backbone2)
#         self.head = Head(self.out_channels_backbone1, num_classes)
#
#     def create_backbone(self, backbone_name):
#         if 'transformer' in backbone_name:
#             if backbone_name == 'swin_transformer_tiny':
#                 backbone = swin_transformer_tiny()
#             elif backbone_name == 'swin_transformer_small':
#                 backbone = swin_transformer_small()
#             else:
#                 backbone = swin_transformer_base()
#             backbone.head = None  # Remove the original classification head
#         elif 'resnet' in backbone_name:
#             if backbone_name == 'resnet18':
#                 backbone = resnet18()
#             elif backbone_name == 'resnet34':
#                 backbone = resnet34()
#             elif backbone_name == 'resnet101':
#                 backbone = resnet101()
#             else:
#                 backbone = resnet50()
#             backbone.fc = None  # Remove the fully connected layer
#         else:
#             raise Exception("Error: wrong backbone name: ", backbone_name)
#         return backbone
#
#     def forward(self, batch_images, cropped_images):
#         initial_features = self.backbone2(batch_images)  # Process full image with backbone 2
#         initial_features = self.global_linear2(initial_features)
#
#         all_cropped_images = []
#         image_indices = []
#         for img_idx, crops in enumerate(cropped_images):
#             merged_crops = torch.cat(crops, dim=0)
#             all_cropped_images.append(merged_crops)
#             image_indices.extend([img_idx] * merged_crops.shape[0])
#
#         all_cropped_images = torch.cat(all_cropped_images, dim=0)
#         image_indices = torch.tensor(image_indices)
#
#         cropped_features = self.backbone1(all_cropped_images)
#         cropped_features = self.global_linear1(cropped_features)
#         cl = self.head(initial_features, cropped_features, image_indices)
#         return cl

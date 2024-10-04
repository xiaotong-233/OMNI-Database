import torch
# import dgl

# # 构建异构图
# def build_graph_from_boxes(boxes):
#     # 假设每个边界框代表一个节点，节点之间的边根据某种准则建立
#     num_boxes = boxes.shape[0]
#     g = dgl.graph(([], []), num_nodes=num_boxes)  # 创建一个空图
#
#     # 添加边的示例，这里需要根据实际情况定义边的连接方式
#     for i in range(num_boxes):
#         for j in range(i + 1, num_boxes):
#             # 假设根据边界框的中心点距离来决定是否连接边
#             center_i = (boxes[i, :2] + boxes[i, 2:]) / 2
#             center_j = (boxes[j, :2] + boxes[j, 2:]) / 2
#             distance = torch.norm(center_i - center_j)
#             if distance < some_threshold:  # some_threshold是一个阈值，根据实际情况设定
#                 g.add_edges(i, j)
#                 g.add_edges(j, i)
#     return g

# 建图


#Used in stage 1 (ANFL)
#用于对图的邻接矩阵进行规范化操作
def normalize_digraph(A):
    b, n, _ = A.shape
    node_degrees = A.detach().sum(dim = -1)
    degs_inv_sqrt = node_degrees ** -0.5
    norm_degs_matrix = torch.eye(n)
    dev = A.get_device()
    if dev >= 0:
        norm_degs_matrix = norm_degs_matrix.to(dev)
    norm_degs_matrix = norm_degs_matrix.view(1, n, n) * degs_inv_sqrt.view(b, n, 1)
    norm_A = torch.bmm(torch.bmm(norm_degs_matrix,A),norm_degs_matrix)
    return norm_A


#Used in stage 2 (MEFL)
#目的是创建一个稀疏矩阵E，用于在MEFL阶段中描述图的边
def create_e_matrix(n):
    end = torch.zeros((n*n,n))
    for i in range(n):
        end[i * n:(i + 1) * n, i] = 1
    start = torch.zeros(n, n)
    for i in range(n):
        start[i, i] = 1
    start = start.repeat(n,1)
    return start,end

# def create_e_matrix(n):
#     end = torch.zeros((n*n, 1), dtype=torch.long)
#     start = torch.zeros((n*n, 1), dtype=torch.long)
#     idx = 0
#     for i in range(n):
#         for j in range(n):
#             start[idx, 0] = i
#             end[idx, 0] = j
#             idx += 1
#     return start, end


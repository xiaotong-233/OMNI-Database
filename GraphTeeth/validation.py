from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from model.MEFL import MEFARG
from dataset import *
from util.utils import *
from conf import get_config, set_logger, set_outdir8, set_env
from model.backbone import fasterrcnn

def get_dataloader(conf):
    print('==> Preparing data...')
    test_annotation_path = ''
    if conf.dataset == 'tooth':
        with open(test_annotation_path) as f:
            test_lines = f.readlines()
        testset = BP4D(test_lines, train=False, val=True)
        test_loader = DataLoader(testset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=GNN_collect_fn)
    return test_loader, len(testset)

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def load_mefarg_model(model_path, num_classes, conf):
    model = MEFARG(num_classes=num_classes, backbone1=conf.arc1, backbone2=conf.arc2)
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
    return model

def validation(net, test_loader, model_path):
    first_stage_model, model_process = fasterrcnn(model_path)
    checkpoint = torch.load(model_path)
    if 'state_dict' in checkpoint:
        first_stage_model.load_state_dict(checkpoint['state_dict'])
    else:
        first_stage_model.load_state_dict(checkpoint)
    first_stage_model.eval()
    neighbor_num = 5
    net.eval()
    all_predictions = []  # 用于存储所有预测结果
    all_boxes = []
    all_targets = []  # 用于存储所有真实标签
    for batch_idx, (inputs, label_tensors) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            processed_outputs, cropped_images = [], []
            for image, target in zip(inputs, label_tensors):
                processed_output, cropped_image = model_process([image], [target], training=False)
                # 判断框的样本数是否小于neighbor_num，如果是，则跳过这个样本
                num_boxes = processed_output[0]['boxes'].size(0)
                if num_boxes < neighbor_num:
                    continue
                processed_outputs.append(processed_output)
                cropped_images.append(cropped_image)
            inputs = inputs.float()
            boxes_batch = [label_tensor[0]['boxes'].clone().detach() for label_tensor in processed_outputs]
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                boxes_batch = [boxes.cuda() for boxes in boxes_batch]
            outputs = net(inputs, cropped_images)
            cl = outputs

            batch_predictions = []
            batch_boxes = []
            for predicted_classes, predicted_boxes in zip(cl, boxes_batch):
                batch_predictions.append(predicted_classes)
                batch_boxes.append(predicted_boxes)
            filtered_indices = [i for i, output in enumerate(processed_outputs) if output[0]['boxes'].size(0) >= neighbor_num]
            all_predictions.extend([batch_predictions[i] for i in filtered_indices])
            all_boxes.extend([batch_boxes[i] for i in filtered_indices])
            all_targets.extend([label_tensors[i] for i in filtered_indices])

    mean_f1, mean_ap, mean_precision, mean_recall, all_f1_scores, all_aps, all_recalls, all_precisions = calc_metrics(all_predictions, all_boxes, all_targets)
    return mean_f1, mean_ap, mean_precision, mean_recall, all_f1_scores, all_aps, all_recalls, all_precisions



def main(conf):
    model_path = ''
    mefarg_model_path = ''  
    num_classes = 10
    test_loader, test_data_num = get_dataloader(conf)
    logging.info("test_data_num: {}".format(test_data_num))
    # net = MEFARG(num_classes=conf.num_classes, backbone1=conf.arc1, backbone2=conf.arc2)
    net = load_mefarg_model(mefarg_model_path, num_classes, conf)
    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()

    # test
    test_mean_f1, test_mean_ap, test_mean_recall, test_mean_precision, test_f1_scores, test_aps, test_recalls, test_precisions = validation(net, test_loader, model_path)

    # log
    infostr = 'test_mean_f1 {:.2f}, test_mean_ap {:.2f}, test_mean_recall {:.2f}, test_mean_precision {:.2f}'.format(
        100. * test_mean_f1,
        100. * test_mean_ap,
        100. * test_mean_recall,
        100. * test_mean_precision,
    )
    logging.info(infostr)

    # 打印详细的列表内容
    logging.info("test_f1_scores: " + ", ".join(["{:.2f}".format(100. * f1) for f1 in test_f1_scores]))
    # logging.info("test_aps: " + ", ".join(["{:.2f}".format(100. * ap) for ap in test_aps]))  # 访问列表
    logging.info("test_recalls: " + ", ".join(["{:.2f}".format(100. * recall) for recall in test_recalls[0.5]]))  # 同上
    logging.info("test_precisions: " + ", ".join(["{:.2f}".format(100. * precision) for precision in test_precisions[0.5]]))  # 同上




# ---------------------------------------------------------------------------------


if __name__=="__main__":
    conf = get_config()
    conf.evaluate = True
    set_env(conf)
    set_outdir8(conf)
    set_logger(conf)
    main(conf)


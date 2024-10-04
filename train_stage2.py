from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
from model.backbone import fasterrcnn
from model.MEFL import MEFARG
from dataset import *
from util.utils import *
from conf import get_config,set_logger,set_outdir2,set_env


def get_dataloader(conf):
    print('==> Preparing data...')
    train_annotation_path = '/mnt/disk1/data0/jxt/dataset/data/resized/list/number_train_data.txt'
    val_annotation_path = '/mnt/disk1/data0/jxt/dataset/data/resized/list/number_val_data.txt'
    if conf.dataset == 'tooth':
        with open(train_annotation_path) as f:
            train_lines = f.readlines()
        with open(val_annotation_path) as f:
            val_lines = f.readlines()
        trainset = BP4D(train_lines, train=True, val=False)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=GNN_collect_fn)
        valset = BP4D(val_lines, train=False, val=True)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=GNN_collect_fn)
    return train_loader, val_loader, len(trainset), len(valset)

def train(conf, net, train_loader, optimizer, epoch, criterion, model_path):
    first_stage_model, model_process = fasterrcnn(model_path)
    checkpoint = torch.load(model_path)
    if 'state_dict' in checkpoint:
        first_stage_model.load_state_dict(checkpoint['state_dict'])
    else:
        first_stage_model.load_state_dict(checkpoint)
    first_stage_model.eval()
    losses = AverageMeter()
    net.train()
    train_loader_len = len(train_loader)
    for batch_idx, (inputs,  targets, relations) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate, batch_idx, train_loader_len)
        targets, relations = targets.float(), relations.long()
        if torch.cuda.is_available():
            inputs, targets, relations = inputs.cuda(), targets.cuda(), relations.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion[0](outputs, targets)
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), inputs.size(0))
    return losses.avg


# Val
def val(net, val_loader, criterion, model_path):
    first_stage_model, model_process = fasterrcnn(model_path)
    checkpoint = torch.load(model_path)
    if 'state_dict' in checkpoint:
        first_stage_model.load_state_dict(checkpoint['state_dict'])
    else:
        first_stage_model.load_state_dict(checkpoint)
    first_stage_model.eval()
    losses = AverageMeter()
    net.eval()
    statistics_list = None
    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        targets = targets.float()
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = net(inputs)
            loss = criterion[0](outputs, targets)
            losses.update(loss.data.item(), inputs.size(0))
            update_list = statistics(outputs, targets.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return losses.avg, mean_f1_score, f1_score_list, mean_acc, acc_list



def main(conf):
    start_epoch = 0
    model_path = 'results/stage1/bs_8_seed_0_lr_0.0001/best_model.pth'
    train_loader, val_loader, train_data_num, val_data_num = get_dataloader(conf)
    num_classes = 18
    train_weight = torch.from_numpy(np.loadtxt(os.path.join(conf.dataset_path, 'list', 'train_weights.txt')))
    logging.info("train_data_num: {} ".format(train_data_num))
    net = MEFARG(num_classes=conf.num_classes, backbone1=conf.arc1, backbone2=conf.arc2)

    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)
    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        train_weight = train_weight.cuda()
    criterion = NodeClassificationLoss(num_classes, train_weight)
    # criterion = [DetectionLoss(weight=train_weight), nn.CrossEntropyLoss()]
    optimizer = optim.AdamW(net.parameters(),  betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    print('the init learning rate is ', conf.learning_rate)

    #train and val
    for epoch in range(start_epoch, conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        print('Start Train')
        train_loss, wa_loss, edge_loss = train(conf, net, train_loader, optimizer, epoch, criterion)
        print('Finish Train' + '\n')
        print('Start Validation')
        val_loss, val_mean_f1, val_mean_ap, val_mean_recall, val_mean_precision = val(net, val_loader, optimizer, epoch, criterion)

        # log
        infostr = 'Epoch:  {}   train_loss: {:.5f}  val_loss: {:.5f}  val_mean_f1 {:.2f},val_mean_ap {:.2f},val_mean_recall {:.2f},val_mean_precision {:.2f}'.format(epoch + 1, train_loss, val_loss, 100. * val_mean_f1, 100. * val_mean_ap, 100. * val_mean_recall, 100. * val_mean_precision)
        logging.info(infostr)

        # save checkpoints
        if (epoch+1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '_model_fold' + str(conf.fold + 1) + '.pth'))

        checkpoint = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(conf['outdir'], 'cur_model_fold' + str(conf.fold + 1) + '.pth'))


# ---------------------------------------------------------------------------------


if __name__=="__main__":
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir2(conf)
    # Set the logger
    set_logger(conf)
    main(conf)

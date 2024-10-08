import os

import torch
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(model_train, model, focal_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()  # 进入训练模式，此时会计算梯度
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets = batch[0], batch[1]  #将批次数据分解为图像和目标两部分
        with torch.no_grad():  # 禁用梯度计算
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #-------------------#
            #   获得预测结果
            #-------------------#
            _, regression, classification, anchors = model_train(images)
            #-------------------#
            #   计算损失
            #-------------------#
            loss_value, _, _ = focal_loss(classification, regression, anchors, targets, cuda = cuda)

            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #-------------------#
                #   获得预测结果
                #-------------------#
                _, regression, classification, anchors = model_train(images)
                #-------------------#
                #   计算损失
                #-------------------#
                loss_value, _, _ = focal_loss(classification, regression, anchors, targets, cuda = cuda)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()  # 计算损失的梯度
            scaler.step(optimizer)  # 更新模型的权重参数
            scaler.update()
            
        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()  # 进入评估模式
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #-------------------#
            #   获得预测结果
            #-------------------#
            _, regression, classification, anchors = model_train(images)
            #-------------------#
            #   计算损失
            #-------------------#
            loss_value, _, _ = focal_loss(classification, regression, anchors, targets, cuda = cuda)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_weights_all10_d0.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_weights_all10_d0.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_weights_all10_d0.pth"))
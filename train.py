#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys


# ## 加载配置

# In[ ]:


from yolo.cfg import read_json, Cfg


# In[ ]:


if sys.argv[-1].startswith('cfg') and sys.argv[-1].endswith('.json'):
    _cfg_path = 'cfg/' + sys.argv[-1]
else:
    _cfg_path = 'cfg/cfg-coco-anchor-B5-res18.json'

print('load cfg from: ' + _cfg_path)


# In[ ]:


cfg = Cfg(read_json(['cfg/cfg-default.json', _cfg_path]))


# In[ ]:


if cfg.anchor.wh is None:
    cfg.anchor['wh'] = cfg.anchor[cfg.base.dataset]
    
if not cfg.train.train:
    # 没有开启训练，则就计算一个epoch
    cfg.train['epoches_num'] = 1

# 检验voc上训练的模型，在coco上的效果
if cfg.mAP.save_voc_to_coco:
    assert cfg.train.train == False
    assert cfg.valid.valid == False
    assert cfg.base.dataset == 'coco'
    assert cfg.net.model_path is not None


# ## 导入依赖

# In[ ]:


sys.path.append("/root/jupyter_workhome/my_packages") 

from tricks.epoches import epoches_n_try as epn


# In[ ]:


import yolo.encoder as en
import yolo.load as load
import yolo.aug as aug
from yolo.backbone import resnet
from yolo.loss import yolo_loss
import yolo.mAP as mAP


# In[ ]:


from torch.utils.data import DataLoader
import cv2
import torch
from tqdm.notebook import tqdm
import os
import time
import random
import json


# ## 创建子目录

# In[ ]:


os.system('rm -rf %s' % cfg.tag)
os.system('mkdir %s' % cfg.tag)

os.system('mkdir %s/coco/' % cfg.tag)


# In[ ]:


# 保存配置
os.system('cp %s %s/' % (_cfg_path, cfg.tag))

with open(cfg.tag+'/cfg_.json', 'w') as f:
    json.dump(cfg.dict, f, ensure_ascii=False, indent=4)


# ## 适配python运行
# python 执行时使用
import logging
LOG_FORMAT = "%(asctime)s [%(levelname)8s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(filename='./%s/log.txt' % cfg.tag,  # 会追加写入
                    level=logging.INFO,
                    format=LOG_FORMAT,
                    datefmt=DATE_FORMAT)

print = logging.info

def tqdm(x):
    return x
# In[ ]:


print(cfg.dict)


# In[ ]:


device = cfg.base.device

device = device if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

print('device: ' + device)


# ## 加载数据

# In[ ]:


if cfg.base.dataset == 'voc':
    sys.path.append("/root/jupyter_workhome/dataset/VOCdevkit/")

    from VocDataset import get_dataset
elif cfg.base.dataset == 'coco':
    sys.path.append("/root/jupyter_workhome/dataset/coco/")
    
    import CocoDataset as coco
else:
    raise RuntimeError('无法识别的数据集')


# In[ ]:


if cfg.base.dataset == 'voc':
    keys = ['file_path', 'w', 'h', 'boxes_conner', 'labels_idx', 'filename']
    
    # todo 应该使用配置文件中的地址
    if cfg.train.train:
        train_data = get_dataset(['2007', '2012'], keys)
        
    if cfg.valid.valid or cfg.mAP.mAP:
        test_data = get_dataset(['2007-test'], keys)


# In[ ]:


if cfg.base.dataset == 'coco':
    keys = ['file_path', 'width', 'height', 'boxes_conner', 'category_id', 'file_name']
    
    if cfg.train.train:
        train_data = coco.CocoDataset(cfg.base.dataset_uri.coco.train.an_paths, 
                                     cfg.base.dataset_uri.coco.train.img_paths, 
                                     keys)
        
    if cfg.valid.valid or cfg.mAP.mAP:
        test_data = coco.CocoDataset(cfg.base.dataset_uri.coco.val.an_paths, 
                                     cfg.base.dataset_uri.coco.val.img_paths, 
                                     keys)


# In[ ]:


def yolo_img_reader(file_path):
    return cv2.imread(file_path)


# In[ ]:


# import random

size_lst = [x*32 for x in range(10, 20)]
cnt = 0
idx = size_lst.index(cfg.net.image_size)


def yolo_aug_train_multi_scale(img, boxes_corner, labels):
    global cnt, idx
    image_size = size_lst[idx]  # 从指定的image_size开始切换
    cnt += 1
    if cnt % (cfg.train.image_size_change_num*cfg.base.batch_size) == 0:  # 每10个batch，切换一次图片尺寸
        idx = random.randint(0, len(size_lst)-1)
    return aug.aug(img, boxes_corner, labels, is_train=True,
                   aug_lst=cfg.train.aug_funcs,
                   image_size=image_size)


# In[ ]:


def yolo_aug_train(img, boxes_corner, labels):
    return aug.aug(img, boxes_corner, labels, is_train=True,
                   aug_lst=cfg.train.aug_funcs,
                   image_size=cfg.net.image_size)


# In[ ]:


def yolo_aug_test(img, boxes_corner, labels):
    return aug.aug(img, boxes_corner, labels, is_train=False,
                   aug_lst=[],
                   image_size=cfg.net.image_size)


# In[ ]:


def yolo_encoder(boxes, classes, S):
    return en.encoder(boxes, classes,
                      grid_num=S,  # 是按批变化的
                      B=cfg.net.B,
                      box_is_corner=True)


# In[ ]:


if cfg.train.multi_scale:
    detection_collate_train = load.DetectionCollate(yolo_img_reader,
                                                    yolo_aug_train_multi_scale,
                                                    yolo_encoder)
else:
    detection_collate_train = load.DetectionCollate(yolo_img_reader,
                                                    yolo_aug_train,
                                                    yolo_encoder)

detection_collate_test = load.DetectionCollate(yolo_img_reader,
                                               yolo_aug_test, 
                                               yolo_encoder)


# In[ ]:


if cfg.train.train:
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.base.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=detection_collate_train
    )


# In[ ]:


if cfg.valid.valid or cfg.mAP.mAP:
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=cfg.base.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=detection_collate_test
    )


# ## 加载模型

# In[ ]:


model = resnet(cfg.net.backbone,
               pretrained=True,
               out_channel=(4+1+cfg.net.C)*cfg.net.B)

model = model.to(device)


# In[ ]:


for name, param in model.named_parameters():
    print(name)


# In[ ]:


def load(model, model_path):
    state_dict_pretrain = torch.load(model_path, map_location=device)
    state_dict = model.state_dict()
    
    for k in state_dict_pretrain.keys():
        if k in state_dict.keys() and (
            state_dict[k].shape == state_dict_pretrain[k].shape
        ):
            # print(k)
            state_dict[k] = state_dict_pretrain[k]
        else:
            print(k)
    model.load_state_dict(state_dict)


# In[ ]:


if cfg.net.model_path is not None:
    print("load model from "+ cfg.net.model_path)
    # model.load_state_dict(torch.load(_model_path, map_location=device))
    load(model, cfg.net.model_path)
else:
    print('train new model')


# ## 优化器&学习率

# In[ ]:


epoch_lr = cfg.train.optimizer.epoch_lr
_epoch_cnt, _lr = epoch_lr[0], epoch_lr[1]


# In[ ]:


if cfg.train.optimizer.use_adam:
    print("use Adam optimizer")
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=_lr[0],
                                 weight_decay=cfg.train.optimizer.weight_decay)
else:
    print("use SGDm optimizer")
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=_lr[0],
                                momentum=cfg.train.optimizer.momentum,
                                weight_decay=cfg.train.optimizer.weight_decay)


# ## 保存

# In[ ]:


# import time

def save(model, loss, best_loss, tag, epoch, save_each=False):

    if loss < best_loss:

        ts = time.strftime("%Y%m%d-%H%M", time.localtime())

        with open('%s/best.txt' % tag, 'a') as f:
            f.write('[%s] epoch:%03d loss:%.4f\n' %
                    (ts, epoch, loss))

        if save_each:
            path = f'./%s/model-%03d-%s-loss_%.4f.pth' % (tag, epoch, ts, loss)
            print(path)
            torch.save(model.state_dict(), path)

        path = f'./%s/model-best.pth' % tag
        print(path)
        torch.save(model.state_dict(), path)

        return loss
    return best_loss


# ## 训练函数

# In[ ]:


def avg_lst(x):
    return round(sum(x) / len(x), cfg.base.csv_decimal)


# In[ ]:


def str_startswith(s, s_list):
    for ss in s_list:
        if s.startswith(ss):
            return True
    return False


def set_parameter_requires_grad(model, model_lock_names: list, requires_grad=False):
    """
    names  在这个列表中的参数，不会被设置
    """
    for name, param in model.named_parameters():
        if str_startswith(name, model_lock_names):
            print(name)
            param.requires_grad = not requires_grad
        else:
            param.requires_grad = requires_grad


# In[ ]:


def lock_shift(model, epoch):
    for k, v in enumerate(cfg.net.model_lock_epoch):
        if epoch == v:
            print('shift model lock at epoch %d with %s' % (v, str(cfg.net.model_lock_names[k])))
            set_parameter_requires_grad(model, model_lock_names=cfg.net.model_lock_names[k])    


# In[ ]:


def lr_shift(epoch):
    for k, v in enumerate(_epoch_cnt):
        if epoch < v:
            return _lr[k]

    return _lr[-1]


# In[ ]:


def check_grad(optimizer):
    for k, p in enumerate(optimizer.param_groups[0]['params']):
        if p.grad is not None and p.grad.isnan().sum().item() > 0:
            print(k, p.shape)
            # print(p)
            # print(p.grad)
            return False
    return True


# In[ ]:


def warm_up(global_step_num, init_lr, end_lr, step_num):
    assert init_lr < end_lr
    if global_step_num > step_num:
        # 返回None，这样后续就不再更新
        return None
    else:
        return init_lr + (end_lr - init_lr)*global_step_num/step_num


# In[ ]:


_init_batch_num = cfg.loss.init_iter_num // cfg.base.batch_size


def train_valid(epoch, model, loader, mode):
    if mode == 'train':
        model.train()

        # 更新学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_shift(epoch)

    else:
        model.eval()  # 只对dropout 和BN有效，不会关闭梯度计算的

    loss_, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6 = [], [], [], [], [], [], []

    d = {}

    # 一个epoch的batch数量
    batches_num = len(loader)

    for k, (imgs, target, _, img_names, wh) in enumerate(tqdm(loader)):

        imgs_ = imgs.to(device)
        target_ = target.to(device)

        # 固定模型的一部分，加快训练速度====================
        if mode == 'train' and cfg.net.model_lock:
            if k == 0:
                lock_shift(model, epoch)

        # 当下是第几次计算(全局的)
        if epoch >= 0:
            global_step_num = batches_num * epoch + k
        else:
            global_step_num = -1

        # 学习率的warm-up====================
        if mode == 'train' and cfg.train.optimizer.warm_up.warm_up:
            new_lr = warm_up(global_step_num,
                             cfg.train.optimizer.warm_up.init_lr,
                             cfg.train.optimizer.warm_up.end_lr,
                             cfg.train.optimizer.warm_up.step_num)
            if new_lr is not None:
                # 更新学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

        # 模型计算=======================================
        pred = model(imgs_)

        # 计算loss======================================
        if cfg.mAP.save_voc_to_coco:  # 因为边框的计算需要
            __anchors_wh = cfg.anchor.voc
        else:
            __anchors_wh = cfg.anchor.wh

        l_t = yolo_loss(pred, target_, global_step_num,
                        B=cfg.net.B, C=cfg.net.C,
                        anchor=cfg.net.anchor,
                        # v1
                        wh_sqrt=cfg.loss.wh_sqrt,
                        # v2
                        anchors_wh=__anchors_wh,
                        anchors_calc_iou=cfg.loss.anchors_calc_iou,
                        init_batch_num=_init_batch_num,
                        # v3
                        conf_target_iou=cfg.loss.conf_target_iou,
                        conf_bce=cfg.loss.conf_bce,
                        cls_ce=cfg.loss.cls_ce,
                        focus_on_small=cfg.loss.focus_on_small)

        l = (l_t * torch.tensor(cfg.loss.coefficients, device=device)).sum()

        # 反向传播======================================
        if mode == 'train':
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            l.backward()

            # 检查梯度是否有nan
            if check_grad(optimizer):
                # Update the parameters with computed gradients.
                optimizer.step()

        # Record the loss and accuracy.
        loss_.append(l.item())
        loss_1.append(l_t[0].item())
        loss_2.append(l_t[1].item())
        loss_3.append(l_t[2].item())
        loss_4.append(l_t[3].item())
        loss_5.append(l_t[4].item())
        loss_6.append(l_t[5].item())

        n = cfg.base.batch_n_print
        if k != 0 and k % n == 0:
            print(f'batches[%5d/%5d]: %.4f (%.4f %.4f %.4f %.4f %.4f %.4f)' % (
                k, len(loader),
                avg_lst(loss_[-n:]),
                avg_lst(loss_1[-n:]),
                avg_lst(loss_2[-n:]),
                avg_lst(loss_3[-n:]),
                avg_lst(loss_4[-n:]),
                avg_lst(loss_5[-n:]),
                avg_lst(loss_6[-n:])))

        del imgs, imgs_
        del target, target_
        del pred, l_t

    d.update(
        {
            mode + "_loss": avg_lst(loss_),
            mode + "_loss_1": avg_lst(loss_1),
            mode + "_loss_2": avg_lst(loss_2),
            mode + "_loss_3": avg_lst(loss_3),
            mode + "_loss_4": avg_lst(loss_4),
            mode + "_loss_5": avg_lst(loss_5),
            mode + "_loss_6": avg_lst(loss_6)
        }
    )
    # 为了降低显存占用
    # del imgs, target, pred, l_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return d


# ## 计算mAP

# In[ ]:


best_mAP = -1


def calc_mAP(model, test_loader, tag, save_coco, epoch):
    global best_mAP

    model.eval()

    preds_dict = None

    for k, (imgs, _1, _2, img_names, wh) in enumerate(tqdm(test_loader)):

        imgs = imgs.to(device)
        wh = wh.to(device)

        pred = model(imgs)

        if cfg.mAP.save_voc_to_coco:
            __anchors_wh = cfg.anchor.voc
        else:
            __anchors_wh = cfg.anchor.wh

        preds_dict = en.flat_decoder(pred, img_names, preds_dict, wh,
                                     c_threshold=cfg.mAP.conf_threshold,
                                     iou_threshold=cfg.mAP.NMS_threshold,
                                     B=cfg.net.B, cls_num=cfg.net.C,
                                     anchor=cfg.net.anchor,
                                     anchors_wh=__anchors_wh,
                                     max_len=cfg.mAP.max_len)
    if cfg.base.dataset == 'voc':
        mAp_dict = mAP.mAP_2007test(preds_dict, use_target_abs=True)

    if cfg.base.dataset == 'coco':
        if cfg.mAP.save_voc_to_coco:
            d = {}
            for k in preds_dict:
                # 将voc的分类id，映射成为coco的分类id
                d[coco.voc20_to_coco80_minus1[k]] = preds_dict[k]
            preds_dict = d

        # 计算mAP
        # mAp_dict = mAP.mAP_coco_val2017(preds_dict, use_target_abs=True)
        target_coco = mAP.load_coco(test_data, use_target_abs=True)
        # mAp_dict = mAP.mAP_coco(preds_dict, target_coco)
        mAp_dict = mAP.mAP(preds_dict, target_coco,
                           threshold=0.5, CLASSES=coco.COCO_CLASSES)
        
        if cfg.mAP.save_voc_to_coco:
            # 只使用voc的20个类计算mAP
            ap_sum = 0
            for i in range(20):
                cls_idx_coco = coco.voc20_to_coco80_minus1[i]
                cls_coco = coco.COCO_CLASSES[cls_idx_coco]
                ap = mAp_dict[cls_coco]
                # 因为预测不到，AP可能就为-1
                ap = ap if ap >=0 else 0
                ap_sum += ap

            mAp_dict['mAP'] = ap_sum/20

    mAp_dict = {
        'AP_'+k: round(mAp_dict[k], cfg.base.csv_decimal) for k in mAp_dict}

    _mAP = mAp_dict['AP_mAP']

    path_lst = []

    if save_coco and _mAP > cfg.mAP.save_coco_when_gt:  # 要不然太大了
        # 保存为coco格式
        path = '%s/coco/coco-mAP%.4f-%d-%s.json' % (tag,
                                                    _mAP,
                                                    epoch,
                                                    time.strftime(
                                                        "%Y%m%d-%H%M", time.localtime())
                                                    )
        print('save to ' + path)
        path_lst.append(path)

    if _mAP > best_mAP:
        path = '%s/coco/coco-best.json' % tag
        print('save to ' + path)
        with open('%s/coco/best.txt' % tag, 'a') as f:
            f.write('[%s] epoch:%03d mAP:%.4f\n' % (time.strftime(
                "%Y%m%d-%H%M", time.localtime()), epoch, _mAP))

        path_lst.append(path)

        best_mAP = _mAP

    if len(path_lst) > 0:
        # 将coco的id从80映射到91
        if cfg.base.dataset == 'coco':
            d = {}
            for cls_idx in preds_dict:
                cls_idx_91 = coco.coco80_minus1_to_91[cls_idx]
                d[cls_idx_91] = preds_dict[cls_idx]
            preds_dict = d
        
        # 保存coco格式的计算结果
        mAP.save_as_coco(preds_dict,
                         path_lst,
                         lambda x: x,
                         lambda x: int(x.split('.')[0]))

        # 这里偷懒一下，直接保存对应的模型
        if cfg.mAP.save_best_mAP_model:
            path = f'./%s/coco/model-best-mAP.pth' % tag
            torch.save(model.state_dict(), path)

    # 为了降低显存占用
    del imgs, _1, _2, wh, pred, preds_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return mAp_dict


# ## train函数

# In[ ]:


best_loss = 9999

@epn(cfg.train.epoches_num,
     func_table_path='./%s/func.csv' % cfg.tag,
     mon_table_path='./%s/monitor.csv' % cfg.tag)
def train(epoch, epoch_num):
    global best_loss

    d = {'lr': optimizer.param_groups[0]['lr']}
    print('lr: %f' % d['lr'])

    if cfg.train.train:
        print(f'[ Train | {epoch + 1:03d}/{epoch_num:03d} ] start')
        d1 = train_valid(epoch, model, train_loader, 'train')
        train_loss = d1['train_loss']
        d.update(d1)
        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{epoch_num:03d} ] loss = {d1['train_loss']:.4f} -> {d1['train_loss_1']:.4f} {d1['train_loss_2']:.4f} {d1['train_loss_3']:.4f} {d1['train_loss_4']:.4f} {d1['train_loss_5']:.4f} {d1['train_loss_6']:.4f}")

    # val & test
    with torch.no_grad():
        if cfg.valid.valid:
            print(f'[ Valid | {epoch + 1:03d}/{epoch_num:03d} ] start')
            d2 = train_valid(-1, model, test_loader, 'val')
            val_loss = d2['val_loss']
            # Print the information.
            d.update(d2)
            if cfg.valid.save_best_loss_model:
                print('try save model')
                best_loss = save(model, val_loss, best_loss, cfg.tag, epoch)
            print(f"[ Valid | {epoch + 1:03d}/{epoch_num:03d} ] loss = {d2['val_loss']:.4f} -> {d2['val_loss_1']:.4f} {d2['val_loss_2']:.4f} {d2['val_loss_3']:.4f} {d2['val_loss_4']:.4f} {d2['val_loss_5']:.4f} {d2['val_loss_6']:.4f}")

        
        # 计算mAP
        if cfg.mAP.mAP:
            print(f'[  mAP  | {epoch + 1:03d}/{epoch_num:03d} ] start')
            d3 = calc_mAP(model, test_loader, cfg.tag, cfg.mAP.save_coco, epoch)
            d.update(d3)
            print(f"[  mAP  | {epoch + 1:03d}/{epoch_num:03d} ] mAP: %s" % str(d3))

    return d


# ## 开始训练

# In[ ]:


train()


# ## 重命名子目录

# In[ ]:


if cfg.base.rename_dir:
    ts = time.strftime("%Y%m%d_%H%M", time.localtime())
    path = '%s-%s' % (cfg.tag, ts)

    os.system('mv %s %s' % (cfg.tag, path))


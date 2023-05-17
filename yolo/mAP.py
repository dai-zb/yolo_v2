#!/usr/bin/env python
# coding: utf-8

from VocDataset import get_dataset, VOC_CLASSES

import boxtool as bt 
from collections_tool import append_dict_dict_list

import numpy as np

import json

###########################################################################
# VOC数据集
__target = None
__target_abs = None  # 绝对尺寸


def load_2007test():
    global __target
    global __target_abs
    if __target is None or __target_abs is None:
        test_data = get_dataset(['2007-test'],
                         ['w', 'h', 'boxes_conner', 'labels_idx', 'filename'])
        for w, h, boxes_conner, labels_idx, filename in test_data:
            for (x1, y1, x2, y2), lab in zip(boxes_conner, labels_idx):
                __target = append_dict_dict_list(lab, filename,
                                                  (x1/w, y1/h, x2/w, y2/h), __target)
                __target_abs = append_dict_dict_list(lab, filename,
                                                   (x1, y1, x2, y2), __target_abs)
        

def mAP_2007test(preds_dict, threshold=0.5, use_target_abs=False):
    load_2007test()
    if use_target_abs:
        return mAP(preds_dict, __target_abs, threshold, VOC_CLASSES)
    else:
        return mAP(preds_dict, __target, threshold, VOC_CLASSES)

###########################################################################
# coco数据集
# __target_coco_val2017 = None
# __target_abs_coco_val2017 = None  # 绝对尺寸
#
#
# def load_coco_val2017():
#     global __target_coco_val2017
#     global __target_abs_coco_val2017
#
#     if __target_coco_val2017 is None or __target_abs_coco_val2017 is None:
#         keys = ['width', 'height', 'boxes_conner', 'category_id', 'file_name']
#
#         data = coco.get_dataset(2017, 'val', keys)
#
#         for w, h, boxes_conner, labels_idx, filename in data:
#             for (x1, y1, x2, y2), lab in zip(boxes_conner, labels_idx):
#                 __target_coco_val2017 = append_dict_dict_list(lab,
#                                                               filename,
#                                                               (x1/w, y1/h, x2/w, y2/h),
#                                                               __target_coco_val2017)
#
#                 __target_abs_coco_val2017 = append_dict_dict_list(lab,
#                                                                   filename,
#                                                                   (x1, y1, x2, y2),
#                                                                   __target_abs_coco_val2017)
#
#
# def mAP_coco_val2017(preds_dict, threshold=0.5,
#                      use_target_abs:'是否使用绝对坐标'=False):
#     load_coco_val2017()
#     if use_target_abs:
#         return mAP(preds_dict,
#                    __target_abs_coco_val2017,
#                    threshold,
#                    coco.COCO_CLASSES)
#     else:
#         return mAP(preds_dict,
#                    __target_coco_val2017,
#                    threshold,
#                    coco.COCO_CLASSES)
###########################################################################
# coco数据集格式的其它数据集

# def load_coco(data:coco.CocoDataset, use_target_abs:'是否使用绝对坐标'):
#     __target_coco = None
#
#     for _, w, h, boxes_conner, labels_idx, filename in data:
#         for (x1, y1, x2, y2), lab in zip(boxes_conner, labels_idx):
#             if use_target_abs:
#                 __target_coco = append_dict_dict_list(lab,
#                                                       filename,
#                                                       (x1, y1, x2, y2),
#                                                       __target_coco)
#             else:
#                 __target_coco = append_dict_dict_list(lab,
#                                                       filename,
#                                                       (x1/w, y1/h, x2/w, y2/h),
#                                                       __target_coco)
#
#     return __target_coco
#
#
# # def mAP_coco(preds_dict, target_coco, threshold=0.5):
# #     return mAP(preds_dict, target_coco, threshold=threshold, CLASSES=coco.COCO_CLASSES)
#
#
###########################################################################
def AP(precision, recall):
    assert len(precision) == len(recall)
    
    # correct ap caculation
    pr = np.concatenate(([0.], precision, [0.]))
    rc = np.concatenate(([0.], recall, [1.]))

    # 这个操作，会将PR曲线左侧的凹陷拉平
    for i in range(pr.size -1, 0, -1):
        pr[i-1] = np.maximum(pr[i-1], pr[i])

    # np.where(cond)[0] 返回符合要求的索引
    idx = np.where(rc[1:] != rc[:-1])[0]
    
    return np.sum((rc[idx + 1] - rc[idx]) * pr[idx + 1])


def PR(pred: '{img_name:[(x1,y1,x2,y2,p),(),()]}',
       target: '{img_name:[(x1,y1,x2,y2),(),()]}}',
       threshold
      ):
    target_num = 0
    
    conf_lst = []
    pred_right_lst = []
    
    img_names = set(pred.keys()).union(set(target.keys()))
    
    for img_name in img_names:
        lst = pred.get(img_name, [])
        # 预测的置信度
        conf = [float(x[-1]) for x in lst]
        # 预测的框
        bboxes_corner_pred = [x[:4] for x in lst]
        # 标注的框
        bboxes_corner_target = target.get(img_name, [])
        
        # 每一个图像标记了多少个框(这个类别下)
        target_num += len(bboxes_corner_target)
        
        # 检查为0的情况
        t_num = len(bboxes_corner_target)
        p_num = len(bboxes_corner_pred)
        
        if t_num == 0 and p_num == 0:
            continue
            
        if t_num == 0 and p_num != 0:
            # 预测的全错
            conf_lst += conf
            pred_right_lst += [0 for _ in range(len(conf))]
            continue
        
        if t_num != 0 and p_num == 0:
            # 都没有预测出来，不需要做任何事情，只要累加一下target_num
            # 前面已经累加了
            continue
        
        # 判断预测的是否正确
        pred_target_map = bt.assign_anchor_to_bbox_corner(bboxes_corner_pred,
                                                          bboxes_corner_target,
                                                          threshold)
        # 预测正确，则对应位标注为索引(从0开始)，否则则标注为-1
        pred_right = (pred_target_map > -1e-6).long().tolist()
        
        conf_lst += conf
        pred_right_lst += pred_right
    
    # 计算 precision & recall
    conf_arr = np.array(conf_lst)
    pred_right_arr = np.array(pred_right_lst)
        
    # 根据置信度重新排序
    sorted_ind = np.argsort(-conf_arr)
    pred_right_arr = pred_right_arr[sorted_ind]
    
    tp_sum = np.cumsum(pred_right_arr)

    recall = tp_sum / target_num if target_num != 0 else np.zeros(len(tp_sum))
    precision = tp_sum / (np.arange(len(tp_sum)) + 1)
    
    return precision, recall


def mAP(preds: '{cls_id:{img_name:[(x1,y1,x2,y2,p),(),()]}}',
             targets: '{cls_id:{img_name:[(x1,y1,x2,y2),(),()]}}',
             threshold,
             CLASSES:'["a", "b", "c", ...]  分类的名称'):
    """
    return {'aa':0.xx, 'bb':0.xx, 'mAP': 0.xx}
              0 表示这个类别有预测，但是全错
             -1 表示这个类别没有预测
    """
    ap_dict = {x: -1 for x in CLASSES}
    ap_dict['mAP'] = -1
    
    if preds is None:  # 模型的输出太多，NMS处理就会太慢，于是就跳过去，所以预测可能为None
        print('preds is None, return AP -1')
        return ap_dict
    
    # 按类计算AP
    for cls_idx in preds:
        if type(cls_idx) is int:
            class_ = CLASSES[cls_idx]
        elif type(cls_idx) is str:
            class_ = cls_idx
            cls_idx = CLASSES.index(class_)
        else:
            raise RuntimeError('idx must be int or str')
        
        pred = preds[cls_idx] # {img_name: [(x1,y1,x2,y2,p),(),()]}

        if len(pred) == 0: #这个类别一个都没有检测到的情况
            print('%s len==0, set AP -1' % class_)
            ap_dict[class_] = -1
            continue
            
        target = targets[cls_idx]
        precision, recall = PR(pred, target, threshold)

        ap_dict[class_] = AP(precision, recall)
    
    # 计算mAP时，只计算目标中有的类
    # 如果预测结果中没有这个类，就会按照0来计算mAP
    cls_cnt = len(list(filter(lambda x: len(targets[x])>0, targets)))
    lst = []
    for k in ap_dict:
        if ap_dict[k] >= 0:
            lst.append(ap_dict[k])
    ap_dict['mAP'] = sum(lst) / cls_cnt
    return ap_dict


def save_as_coco(preds_dict, path_lst,
                 cls_func,
                 id_func,
                 round_cnt=4):
    """
    preds_dict {"cls_idx": {"img_name":[(x1, y1, x2, y2, p), (), ()]}}
    注意，coco格式中，x1 y1 x2 y2都是绝对坐标
    
    path_lst  保存的json格式的路径 列表
    
    cls_func  函数 判断将cls转为编号
    id_func   函数 根据图片名称转为id
    
    round_cnt  保存位数
    """
    
    img_lst = []
    cls_lst = []
    
    for idx in preds_dict:
        cls_lst.append(idx)
        pred = preds_dict[idx]
        for img_name in pred:
            img_lst.append(img_name)
    
    # 去重，否则保存的文件中有重复
    img_lst = set(img_lst)
    cls_lst = set(cls_lst)
    
    coco_lst = []
    for img_name in img_lst:   # 遍历文件
        for cls in cls_lst:    # 遍历类
            lst = preds_dict[cls][img_name]
            if len(lst) != 0:
                for item in lst:  # 遍历box
                    bbox = item[:4]
                    bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
                    bbox = [round(x, round_cnt) for x in bbox]
                    d = {
                        'image_id': id_func(img_name),
                        'category_id': cls_func(cls),
                        'bbox': bbox,
                        "score": item[4]
                    }
                    coco_lst.append(d)

    for path in path_lst:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(coco_lst, f, ensure_ascii=False, indent=2) # indent表示空格缩进的长度


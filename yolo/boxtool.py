#!/usr/bin/python3
# coding=utf-8

import torch


def __to_tensor(x):
    if type(x) is not torch.Tensor:
        return torch.tensor(x)
    return x


def box_iou_cross_corner(boxes1, boxes2):
    """
    计算两组框的交并比(IOU inter over union)
                 n0          n1          n2
      m0  iou(m0,n0)  iou(m0,n1)  iou(m0,n2)
      m1  iou(m1,n0)  iou(m1,n1)  iou(m1,n2)
      m2  iou(m2,n0)  iou(m2,n1)  iou(m2,n2)

    :param boxes1:  Size([m, 4])  (对角表示)
    :param boxes2:  Size([n, 4])  (对角表示)
    :return:  Size([m, n])
    """
    boxes1 = __to_tensor(boxes1)
    boxes2 = __to_tensor(boxes2)
    
    boxes1 = boxes1.reshape(-1, 4)
    boxes2 = boxes2.reshape(-1, 4)

    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))

    areas1 = box_area(boxes1)  # shape 是 (boxes1的数量,)
    areas2 = box_area(boxes2)  # shape 是 (boxes2的数量,)

    # 计算出交点左上角的坐标
    inter_upper_lefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lower_rights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    # 左上 - 右下 即为相交边框的尺寸
    # 如果两者不相交，则相减即为负，此时只需要将其置为0
    inters = (inter_lower_rights - inter_upper_lefts).clamp(min=0)

    # `inter_areas` 与 `union_areas`的形状: (boxes1的数量, boxes2的数量)，其中元素是浮点型，表示IOU的值
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas

    iou = inter_areas / union_areas

    return torch.where(torch.isnan(iou), torch.full_like(iou, 0), iou)


def box_iou_cross_center(boxes1, boxes2):
    """
    计算两组框的交并比(IOU inter over union)
                 n0          n1          n2
      m0  iou(m0,n0)  iou(m0,n1)  iou(m0,n2)
      m1  iou(m1,n0)  iou(m1,n1)  iou(m1,n2)
      m2  iou(m2,n0)  iou(m2,n1)  iou(m2,n2)

    :param boxes1:  Size([m, 4])  (中心表示)
    :param boxes2:  Size([n, 4])  (中心表示)
    :return:  Size([m, n])
    """
    boxes1 = center_to_corner(boxes1)
    boxes2 = center_to_corner(boxes2)

    return box_iou_cross_corner(boxes1, boxes2)


def box_iou_corner(boxes1, boxes2):
    boxes1 = __to_tensor(boxes1)
    boxes2 = __to_tensor(boxes2)
    
    shape1 = boxes1.shape
    shape2 = boxes2.shape

    assert shape1 == shape2, '传入的张量必须同形'
    assert shape1[-1] == 4, '最后一轴的长度必须是4'
    assert shape2[-1] == 4, '最后一轴的长度必须是4'

    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))

    boxes1 = boxes1.reshape(-1, 4)
    boxes2 = boxes2.reshape(-1, 4)

    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)

    # 计算出交点左上角的坐标
    inter_upper_lefts = torch.max(boxes1[:, :2], boxes2[:, :2])
    inter_lower_rights = torch.min(boxes1[:, 2:], boxes2[:, 2:])

    # 左上 - 右下 即为相交边框的尺寸
    # 如果两者不相交，则相减即为负，此时只需要将其置为0
    inters = (inter_lower_rights - inter_upper_lefts).clamp(min=0)

    inter_areas = inters[:, 0] * inters[:, 1]
    union_areas = areas1 + areas2 - inter_areas

    # 在某些极端情况下，union 为0，此时结果就为nan，如此替换为0
    iou = (inter_areas / union_areas).reshape(shape1[:-1])

    return torch.where(torch.isnan(iou), torch.full_like(iou, 0), iou)


def box_iou_center(boxes1, boxes2):
    boxes1 = center_to_corner(boxes1)
    boxes2 = center_to_corner(boxes2)

    return box_iou_corner(boxes1, boxes2)


def corner_to_center(corner):
    """
    左上x, 左上y, 右下x, 右下y  的边界框的表示 (这种表示方便计算框的面积，标注中常见这种表示)
    转换为
    x中心, y中心, w, h 的边界框的表示  (这种表示方便计算锚框的位置)
    :param corner:   Size[L, 4]
    :return:     Size[L, 4]，每一行代表一个坐标
    """
    corner = __to_tensor(corner)
    
    s = corner.shape
    corner = corner.reshape(-1, 4)

    x1, y1, x2, y2 = corner[:, 0], corner[:, 1], corner[:, 2], corner[:, 3]

    x = (x2 + x1) * 0.5
    y = (y2 + y1) * 0.5
    w = x2 - x1
    h = y2 - y1

    return torch.stack([x, y, w.float(), h.float()], dim=1).reshape(s)


def center_to_corner(center):
    """
    x中心, y中心, w, h 的边界框的表示  (这种表示方便计算锚框的位置)
    转换为
    左上x, 左上y, 右下x, 右下y  的边界框的表示 (这种表示方便计算框的面积，标注中常见这种表示)

    :param center:   Size[L, 4]
    :return:     Size[L, 4]，每一行代表一个坐标
    """
    center = __to_tensor(center)
    
    s = center.shape
    center = center.reshape(-1, 4)

    center_x, center_y, w, h = center[:, 0], center[:, 1], center[:, 2], center[:, 3]
    half_w = w.reshape(-1) * 0.5
    half_h = h.reshape(-1) * 0.5
    c_x = center_x.reshape(-1)
    c_y = center_y.reshape(-1)
    x1 = c_x - half_w
    y1 = c_y - half_h
    x2 = c_x + half_w
    y2 = c_y + half_h

    return torch.stack([x1, y1, x2, y2], dim=1).reshape(s)


def nms_corner(boxes, scores, iou_threshold=0.5):
    """
    根据boxes之间的IOU和分数(分类概率)，筛选出要保留的框的索引
    :param boxes:     Size([m, 4]) 锚框，对角表示  为分类预测不为背景的框
    :param scores:    Size([m])    一个框对应一个分数
    :param iou_threshold:   合并时的IOU阈值
    :return:   Size([n])    返回的是一个索引
            Size([n, 4])  返回索引后的box
    """
    boxes = __to_tensor(boxes)
    scores = __to_tensor(scores)

    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1)

    assert boxes.shape[0] == scores.shape[0], 'boxes与scores必须匹配'

    b = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留预测边界框的指标
    while b.numel() > 0:
        keep.append(b[0])
        if b.numel() == 1:
            break
        # 排在第一个框跟其余的框求IOU
        iou = box_iou_cross_corner(boxes[b[0], :].reshape(1, 4), 
                                   boxes[b[1:], :].reshape(-1, 4)).reshape(-1)

        # 计算出所有IOU小于阈值的索引
        inds = torch.nonzero((iou <= iou_threshold).long()).reshape(-1)

        if len(inds) == 0:
            break

        # 因为求IOU时，使用了b[1:]切片，所以inds要加一
        inds = inds + 1

        # 选取其余的框，重复上述过程
        b = b[inds]
    idx = torch.tensor(keep, device=boxes.device).reshape(-1)

    return idx, boxes[idx], scores[idx]


def nms_center(boxes, scores, iou_threshold=0.5):
    boxes = __to_tensor(boxes)
    scores = __to_tensor(scores)
    
    boxes = center_to_corner(boxes)
    idx, boxes, scores = nms_corner(boxes, scores, iou_threshold)
    boxes = corner_to_center(boxes)
    return idx, boxes, scores


# 感觉这个函数用不到，有些鸡肋
# def nms_by_label_corner(bboxes, scores, label, iou_threshold: float = 0.5):
#     """
#     bboxes  对角表示,以list的格式输入

#     注意，NMS只能一帧图像一帧图像的进行，无法进行batch操作
#     """
#     if isinstance(bboxes, torch.Tensor):
#         bboxes = bboxes.tolist()
#     if isinstance(scores, torch.Tensor):
#         scores = scores.tolist()
#     if isinstance(label, torch.Tensor):
#         label = label.tolist()

#     # 根据label进行分组
#     dic = {}

#     for b, s, l in zip(bboxes, scores, label):
#         lst = dic.get(l, [])
#         lst.append([b[0], b[1], b[2], b[3], s])
#         dic[l] = lst

#     # 各组分别进行NMS
#     ret_bboxes_lst = []
#     ret_scores_lst = []
#     ret_label_lst = []
#     for k in dic:
#         t = torch.tensor(dic[k])

#         box = t[:, :4]
#         score = t[:, 4]

#         _, box, score = nms_corner(box, score, iou_threshold)

#         box = box.tolist()

#         score = score.tolist()

#         lst = [k for x in box]

#         ret_bboxes_lst += box
#         ret_scores_lst += score
#         ret_label_lst += lst

#     return ret_bboxes_lst, ret_scores_lst, ret_label_lst


def assign_anchor_to_bbox_corner(anchors, bounding_boxes, iou_threshold=0.5):
    """
    将最接近的真实边界框分配给锚框
    :param anchors:   多个锚框 (对角表示)框的表示格式是左上x 左上y 右下x 右下y 
                      [[x1 y1 x2 y2] [x1 y1 x2 y2] ... ] Size([n, 4])
    :param bounding_boxes:  多个边界框  (对角表示)框的表示格式是左上x 左上y 右下x 右下y
                      [[x1 y1 x2 y2] [x1 y1 x2 y2] ... ]  Size([m, 4])
    :param iou_threshold:   IOU的阈值，超过这个值的，会标注相应的锚框为边界框的编号
    
    :return:    numpy数组或者torch的一维tensor  Size([n,])，其中值为box_idx或-1(表示没有分配)
    """
    anchors = __to_tensor(anchors)
    bounding_boxes = __to_tensor(bounding_boxes)
    
    anchors_num = anchors.shape[0]
    bounding_boxes_num = bounding_boxes.shape[0]

    device = anchors.device

    # 位于第i⾏和第j列的元素 x_ij 是锚框i和真实边界框j的IoU
    jaccard = box_iou_cross_corner(anchors, bounding_boxes)

    # 对于每个锚框，分配的真实边界框的编号(对应输入anchors的索引)
    # 没有分配锚框的边界框就标记为-1
    anchors_bbox_map = torch.full([anchors_num], -1, dtype=torch.long, device=device)

    # 根据阈值，决定是否分配真实边界框
    # 全为-1的一列，后续用于覆盖jaccard，可以表示将这一列移除
    col_discard = torch.full([anchors_num], -1, dtype=torch.long, device=device)

    # 全为-1的一行，后续用于覆盖jaccard，可以表示将这一行移除
    row_discard = torch.full([bounding_boxes_num], -1, dtype=torch.long, device=device)

    # IOU都是大于0的

    for _ in range(bounding_boxes_num):
        max_idx = torch.argmax(jaccard).item()
        max_value = torch.max(jaccard).item()
        if max_value < iou_threshold:
            # 最大值都小于阈值，则不再继续
            break

        # 由于argmax返回的是一维的索引，这里将其转换成为二维的坐标
        box_idx = max_idx % bounding_boxes_num
        anc_idx = max_idx // bounding_boxes_num

        # 标记一下最大IOU对应的锚框
        anchors_bbox_map[anc_idx] = box_idx

        # 遮挡jaccard中列，这样等价于将这些列移除
        jaccard[:, box_idx] = col_discard
        # 遮挡jaccard中行，这样等价于将这些行移除
        jaccard[anc_idx, :] = row_discard

    return anchors_bbox_map

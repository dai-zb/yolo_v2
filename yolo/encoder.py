# encoding:utf-8

import boxtool as bt

import torch
import torch.nn.functional as F

from collections_tool import append_dict_dict_list


def calc_yolo_anchor_corner(grid_num: int,
                            anchors_wh: "anchor的宽高"):
    B_, _2 = anchors_wh.shape
    assert _2 == 2
    boxes = torch.zeros([1, grid_num, grid_num, B_, 4], device=anchors_wh.device)
    boxes[..., :2] = 0.5
    
    # 这里的计算比较固定，为了减少计算量，就不调用to_real_box_corner
    # to_real_box_corner 有指数运算
    """
    return to_real_box_corner(boxes,
                              B_,
                              wh_exp=True,
                              anchors_wh=anchors_wh)
    """
    x = boxes[:, :, :, :, 0]
    y = boxes[:, :, :, :, 1]

    a = torch.arange(grid_num, device=anchors_wh.device)
    # 中心表示
    boxes[:, :, :, :, 0] = (x + a[:, None]) / grid_num
    boxes[:, :, :, :, 1] = (y + a[:, None, None]) / grid_num
    boxes[:, :, :, :, 2:] = anchors_wh[None, None, None, :, :]

    return bt.center_to_corner(boxes)


def to_yolo_box_center(boxes: torch.Tensor,
                       grid_num: int,
                       wh_exp: bool,
                       anchors_wh: "anchor的宽高"):
    """
    boxes Size(batch_size, S, S, B, 4) 中心表示，都是相对坐标
                    [b_x, b_y, b_w, b_h]
    grid_num
    
    wh_exp    是否要对wh进行exp(逆)操作
    
    x,y 变成相当于cell左上角坐标的相对距离(并除以cell的宽高)
    w,h 则根据yolo v2的格式进行计算
    """
    Batch_size_, S1_, S2_, B_, _ = boxes.shape
    assert S1_ == S2_
    assert S1_ == grid_num

    boxes = boxes.clone()
    x = boxes[:, :, :, :, 0] * grid_num
    y = boxes[:, :, :, :, 1] * grid_num

    a = torch.arange(grid_num, device=boxes.device)
    # 中心表示
    x = x - a[:, None]
    boxes[:, :, :, :, 0] = x
    y = y - a[:, None, None]
    boxes[:, :, :, :, 1] = y

    if wh_exp:
        wh = boxes[:, :, :, :, 2:]
        wh = torch.log(
            (wh / anchors_wh[None, None, None, :, :]).clamp(min=1e-6))  # log 0 会出问题
        boxes[:, :, :, :, 2:] = wh

    return boxes  # t_x, t_y, t_w, t_h


def encoder(boxes, classes,
            grid_num, B,
            box_is_corner=True):
    """
    boxes  Size(-1, 4)  默认对角表示，且是相对与图片尺寸的相对坐标
    cat 结果是否要合并(合并便于计算)
    box_is_corner   是否是对角表示
    """

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)
    boxes = boxes.view(-1, 4)

    if not isinstance(classes, torch.Tensor):
        classes = torch.tensor(classes)

    # 如果输入的box是对角表示，则转为中心表示(根据中心确定分到哪个cell)
    if box_is_corner:
        boxes = bt.corner_to_center(boxes)

    # 计算bbox的中心点所在的坐标
    idx = (boxes[:, :2] * grid_num).floor().long()

    # class 不再转为one-hot，计算loss时再转
    # one_hot = F.one_hot(classes, cls_num)

    # YOLO中，box采用的是中心表示
    box_tensor = torch.zeros((grid_num, grid_num, 4))
    # cls_tensor = torch.zeros((grid_num, grid_num, cls_num))
    cls_tensor = torch.full([grid_num, grid_num, 1], -1)  # 没有物体的分类编号为-1
    mask = torch.zeros((grid_num, grid_num), dtype=torch.int32)

    for k, i in enumerate(idx):
        x, y = i.tolist()

        # 限制范围(在coco上，有概率超范围)
        if x == -1:
            print('encoder.py: 120, x == %d' % x)
            x = 0
        if x == grid_num:
            print('encoder.py: 123, x == %d' % x)
            x = grid_num - 1
        if y == -1:
            print('encoder.py: 126, y == %d' % y)
            y = 0
        if y == grid_num:
            print('encoder.py: 129, y == %d' % y)
            y = grid_num - 1

        # 因为是按行存储的(行是最后一维)，所有索引时是 y, x
        mask[y, x] = 1

        # 相对于网格的左上角的偏移(这样可以配合使用Sigmoid函数)
        box_tensor[y, x] = boxes[k]

        # cls_tensor[y, x] = one_hot[k]
        cls_tensor[y, x] = classes[k]

    box_tensor = torch.cat((box_tensor, mask[:, :, None].float(), cls_tensor), dim=-1)
    # 重复了B次
    box_tensor = box_tensor.repeat(1, 1, B).reshape(grid_num, grid_num, -1)
    return box_tensor, mask


def reshape_and_split(x, grid_num, B, cls_num):
    """
    x           shape (batch_size, S, S, B*(4 + 1 + C))
    
    boxes       shape (batch_size, S, S, B*4)
    confidence  shape (batch_size, S, S, B)
    classes     shape (batch_size, S, S, B*C)
    """
    batch_size = x.shape[0]

    x = x.reshape(batch_size, grid_num, grid_num, -1)

    assert x.shape[3] == B * (5 + cls_num), (f'%d , channel != %d' % (B * (5 + cls_num), x.shape[3]))

    boxes = []
    confidence = []
    classes = []

    for i in range(B):
        boxes.append(x[:, :, :, i * (5 + cls_num):(i * (5 + cls_num) + 4)])
        confidence.append(x[:, :, :, i * (5 + cls_num) + 4])
        classes.append(x[:, :, :, (i * (5 + cls_num) + 5):((i + 1) * (5 + cls_num))])

    boxes = torch.stack(boxes, dim=-2)  # boxes的轴数为4
    confidence = torch.stack(confidence, dim=-1)  # confidence的轴数为3
    classes = torch.stack(classes, dim=-2)

    return boxes, confidence, classes


def reshape_and_split_clsidx(x, grid_num, B):
    """
    返回的就不是one hot格式的了
    """
    return reshape_and_split(x, grid_num, B, 1)


def __calc_mask(confidence, c_threshold, keep_max):
    """
    计算出哪些bbox含有物体
    """

    mask1 = confidence > c_threshold

    if keep_max:
        # 计算出每一个batch(即每一个图片)的最大值对应的坐标，并转为mask
        # 这样可以防止空的预测
        b, c, h, w = confidence.shape
        idx = confidence.view(b, -1).max(dim=-1)[1]
        mask2 = F.one_hot(idx, c * h * w).reshape(b, c, h, w)

        return ((mask1 + mask2) > 0)
    else:
        return mask1


def __calc_classes(x):
    """
    根据网络输出的结果，计算对应的概率和分类
    """

    # 计算出最大概率对应的类别
    t_max = torch.max(x, dim=-1)
    cls_prob, cls_idx = t_max[0], t_max[1]

    return cls_prob, cls_idx


def to_real_box_corner(boxes,
                       grid_num,
                       wh_exp: bool,
                       anchors_wh: "anchor的宽高"):
    """
    boxes Size(Batch_size, S, S, B, 4)
    
    wh_exp    是否要对wh进行exp操作
    """
    Batch_size_, S1_, S2_, B_, _ = boxes.shape
    assert S1_ == S2_
    assert S1_ == grid_num
    boxes = boxes.clone()

    x = boxes[:, :, :, :, 0]
    y = boxes[:, :, :, :, 1]

    a = torch.arange(grid_num, device=boxes.device)
    # 中心表示
    x = (x + a[:, None]) / grid_num
    y = (y + a[:, None, None]) / grid_num

    boxes[:, :, :, :, 0] = x
    boxes[:, :, :, :, 1] = y

    if wh_exp:
        wh = boxes[:, :, :, :, 2:]
        wh = anchors_wh[None, None, None, :, :] * torch.exp(wh)
        boxes[:, :, :, :, 2:] = wh

    return bt.center_to_corner(boxes)


def decoder(x, wh=None,
            c_threshold=0.1,
            B=5, cls_num=20, keep_max=True,
            anchor: bool = True,
            anchors_wh: "anchor的宽高" = None):
    """
    x  Tensor  Size(Batch_size, S, S, B*(5+C))
    
    wh is not None，则返回的是真实边框的尺寸，否则返回相对尺寸
    wh Size(Batch_size, 2)
    
    wh_sigmoid  是否要对wh进行sigmoid计算
    wh_exp    是否要对wh进行exp操作
    
    return  [[boxes(N, 4), conf(N), cls(N)], ..., ] len()=Batch_size
    """

    Batch_size, S1, S2, _L = x.shape
    assert S1 == S2
    assert _L == B * (5 + cls_num)
    grid_num = S1
    
    if anchor and not isinstance(anchors_wh, torch.Tensor):
        anchors_wh = torch.tensor(anchors_wh, device=x.device)

    # 将网络的输出进行拆分
    boxes, confidence, classes = reshape_and_split(x, grid_num, B, cls_num)
    if anchor:
        # xy
        boxes[:, :, :, :, :2] = torch.sigmoid(boxes[:, :, :, :, :2])
    else:
        # xywh
        boxes = torch.sigmoid(boxes)
        
    confidence = torch.sigmoid(confidence)
    classes = torch.sigmoid(classes)

    # 将相对与grid的xy表示，转为相对于全图，并转为对角表示
    boxes_corner = to_real_box_corner(boxes, grid_num, anchor, anchors_wh)
    
    # 限制范围
    boxes_corner = boxes_corner.clamp(min=0, max=1)

    if wh is not None:
        boxes_corner = boxes_corner * torch.cat([wh, wh],
                                                dim=-1)[:, None, None, None, :]

    # 计算cell对应的概率和分类
    cls_prob, cls_idx = __calc_classes(classes)

    confidence = confidence * cls_prob  # Pr(Object)*Pr(Class|Object)

    # 根据阈值判断哪个框是有目标的
    mask = __calc_mask(confidence, c_threshold, keep_max)

    lst = []

    # 按batch进行遍历
    for idx, box, conf, cls_i in zip(mask, boxes_corner, confidence, cls_idx):
        # idx就是每一个batch中有目标的bbox的索引
        box = box[idx]
        conf = conf[idx]
        cls_i = cls_i[idx]

        lst.append((box, conf, cls_i))  # 边框 分类概率

    return lst


def flat_decoder(x, img_names, preds_dict=None, wh=None,
                 c_threshold=0.1, iou_threshold=0.5,
                 B=5, cls_num=20, keep_max=True,
                 anchor: bool = True,
                 anchors_wh: "anchor的宽高" = None,
                 max_len=1000):
    """
    x  Tensor  Size(Batch_size, S, S, B, B*(5+C))
    
    return {"cls_idx": {"name":[(x1, y1, x2, y2, p), (), ()]}}
    
    wh is not None，则返回的是真实边框的尺寸，否则返回相对尺寸
    wh Size(Batch_size, 2)
    
    max_len  每个图片最多有多少个预测结果，超过的，就直接跳过处理
    """
    # 先进行解码
    bpc_lst = decoder(x, wh,
                      c_threshold,
                      B, cls_num,
                      keep_max,
                      anchor, anchors_wh)
    """
    bpc_lst   [(boxes_corner, probs, cls_indexs)]*batch_size
    """

    for (boxes_corner, probs, cls_indexs), name in zip(bpc_lst, img_names):

        if len(probs) > max_len:
            # print(len(probs)) 随机模型大约在 100到200之间
            # 返回太多结果，做NMS会很慢
            continue

        # NMS
        idx, boxes_corner, probs = bt.nms_corner(boxes_corner, probs, iou_threshold)
        cls_indexs = cls_indexs[idx]

        boxes_corner = boxes_corner.tolist()
        probs = probs.tolist()
        cls_indexs = cls_indexs.tolist()

        for box_corner, prob, cls_index in zip(boxes_corner, probs, cls_indexs):
            x1, y1, x2, y2 = box_corner

            preds_dict = append_dict_dict_list(cls_index,
                                               name,
                                               (x1, y1, x2, y2, prob),
                                               preds_dict)

    return preds_dict

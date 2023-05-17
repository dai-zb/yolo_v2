# coding: utf-8
import encoder as en
import boxtool as bt

import torch
import torch.nn as nn
import torch.nn.functional as F


def yolo_loss(pred, target, global_step_num:'当下是第几次计算(全局的)',
              B=2, C=20,
              # v1 or v2
              anchor: bool = False,
              # v1
              wh_sqrt: bool = True,
              # v2
              anchors_wh: list = None,
              anchors_calc_iou: bool = True,
              init_batch_num: int = -1,
              # v3
              conf_target_iou: bool = True,
              conf_bce: bool = False,
              cls_ce: bool = False,
              focus_on_small: bool = False
              ):
    """
    v1
      # 不使用anchor
      anchor = False
      # 计算宽高误差开根号
      wh_sqrt = True
    v2
      # 使用anchor
      anchor = True
      # 使用的anchor的宽高
      anchors_wh = [(w, h), (w, h), (w, h), (w, h), (w, h)]
      # 使用anchor计算iou，这个的存在是为了兼容之前的代码
      anchors_calc_iou = True
      # 对初始的box预测限制，使其接近于anchor
      init_batch_num = 12800
    v3
      # 置信度的目标，使用1，而不是IOU
      conf_target_iou = False
      # 计算置信度损失时，使用二元交叉熵
      conf_bce = True
      # 计算分类损失时，使用交叉熵
      cls_ce = True
      # 对小物体使用更高的权重
      focus_on_small = False
    """
    # assert pred.shape == target.shape, 'pred.shape != target.shape'
    N, S1, S2, L_ = pred.shape
    assert S1 == S2 and L_ == B * (5 + C)
    N_t, S1_t, S2_t, L_t = target.shape
    assert N == N_t and S1 == S1_t and S2 == S2_t
    assert L_t == B * (5 + 1)
    S = S1

    if not anchor:
        anchors_wh = None
        init_batch_num = -1
    else:
        wh_sqrt = None
        if not isinstance(anchors_wh, torch.Tensor):
            # 将anchor的宽高转为tensor
            anchors_wh = torch.tensor(anchors_wh, device=pred.device)

    ###################################################
    # 处理预测值
    boxes_pred, confidence_pred, class_pred = en.reshape_and_split(pred, S, B, C)

    if not anchor:
        boxes_pred = torch.sigmoid(boxes_pred)
    else:
        # 使用anchor时，只对xy计算sigmoid
        boxes_pred[:, :, :, :, :2] = torch.sigmoid(boxes_pred[:, :, :, :, :2])

    if not conf_bce:
        # 因为 BCEWithLogitsLoss 中会计算sigmoid
        # 置信度
        confidence_pred = torch.sigmoid(confidence_pred)

    if not cls_ce:
        # 因为 CrossEntropyLoss 中会使用softmax
        # 分类
        class_pred = torch.sigmoid(class_pred)

    #################################################
    # 处理目标值
    with torch.no_grad():
        boxes_target_real_center, confidence_target, class_target_idx = en.reshape_and_split_clsidx(target, S, B)
        # boxes_target_real_center 是中心表示的
        boxes_target = en.to_yolo_box_center(boxes_target_real_center, S,
                                             wh_exp=anchor,  # 使用anchor，则使用 wh_exp
                                             anchors_wh=anchors_wh  # 传入锚框的宽高
                                             )
        # 转为对角表示，因为要计算iou
        boxes_target_real = bt.center_to_corner(boxes_target_real_center)

    # target中置信度大于0，即标记这个grid cell中有物体  (I_obj_i)
    mask = confidence_target[:, :, :, 0] > 0  # 含有obj的grid cell的mask

    ######################
    # 筛选出负责预测&不负责预测的cell
    
    # 注意，这里有detach()
    # 使用预测值计算iou
    boxes_pred_real = en.to_real_box_corner(boxes_pred, S,
                                    wh_exp=anchor,
                                    anchors_wh=anchors_wh)
    iou_p = bt.box_iou_corner(boxes_pred_real, boxes_target_real).detach()

    if anchor and anchors_calc_iou:
        # 使用anchor计算iou
        anchor_xyxy = en.calc_yolo_anchor_corner(S, anchors_wh)
        anchor_xyxy = anchor_xyxy.repeat(N, 1, 1, 1, 1)
        # anchor，使用anchor box与gt box求iou
        iou = bt.box_iou_corner(anchor_xyxy, boxes_target_real).detach()
    else:
        # 使用预测值计算iou
        iou = iou_p
    
    # 一半的框为True，一半的框为False
    max_iou, max_idx = iou.max(dim=-1)

    # 负责检查物体的box
    # cell中要包含物体，且这个box与物体框的IOU比同cell的其它box都大
    onehot = F.one_hot(max_idx, B).bool()
    I_obj_ij = onehot * mask[:, :, :, None]  # 总和与物体(标签)的数量相同
    I_nobj_ij = ~I_obj_ij
    # assert I_obj_ij.long().sum().item() + I_nobj_ij.long().sum().item() == S*S*B
    # assert I_obj_ij.shape == I_nobj_ij.shape
    # assert I_obj_ij.shape[:-1] == mask.shape

    # 筛选出(有物体的cell中)负责预测物体的那一半(1/B)的boxes
    box_pred_response = boxes_pred[I_obj_ij].view(-1, 4)
    box_target_response = boxes_target[I_obj_ij].view(-1, 4)

    # 筛选出(有物体的cell中)负责预测物体的那一半(1/B)的框对应的置信度
    confidence_pred_response = confidence_pred[I_obj_ij].view(-1)

    if conf_target_iou:
        # 置信度的标签值，则是使用最大的iou表示
        # confidence_target_response = max_iou[mask].view(-1)
        confidence_target_response = iou_p[I_obj_ij].view(-1)
    else:
        # v3版本，使用的是1为置信度目标
        confidence_target_response = torch.ones_like(confidence_pred_response,
                                                     device=pred.device)

    # 筛选出(有物体的cell中)不负责预测物体的那一半(1-1/B)的框对应的置信度  
    confidence_pred_no_response = confidence_pred[I_nobj_ij].view(-1)
    # 置信度的标签值(不负责预测),则用0来表示
    confidence_target_no_response = torch.zeros_like(confidence_pred_no_response,
                                                     device=pred.device)

    # 筛选出(有物体的cell中)负责预测物体的那一半(1/B)的class
    class_pred_response = class_pred[I_obj_ij].view(-1, C)
    class_target_response = class_target_idx[I_obj_ij].view(-1).long()
    if not cls_ce:
        # 不使用交叉熵计算分类误差，则需要转为one-hot
        class_target_response = F.one_hot(class_target_response,
                                          C).float()

    #################################################
    # 计算loss
    # 定位误差的系数(v3引入)
    if focus_on_small:
        with torch.no_grad():  # 因为只是计算一个梯度，所以就不计算梯度了
            # 使用的是中心表示，因为要获取宽高
            box_target_real_res = boxes_target_real_center[I_obj_ij].view(-1, 4)
            box_target_real_res = box_target_real_res.detach()
            s_wh = 2 - box_target_real_res[:, 2] * box_target_real_res[:, 3]
            s_wh = s_wh[:, None]
    else:
        s_wh = 1

    # loss 1 中心定位误差
    loss1 = F.mse_loss(box_pred_response[:, :2],
                       box_target_response[:, :2],
                       reduction='none')
    loss1 = (s_wh * loss1).sum() / N

    # loss 2 宽高定位误差
    if wh_sqrt:
        # v1 版本的loss，需要开方
        loss2 = F.mse_loss(torch.sqrt(box_pred_response[:, 2:].clamp(min=1e-6)),
                           torch.sqrt(box_target_response[:, 2:].clamp(min=1e-6)),
                           reduction='none')
        # 注意，这里限制了最小值，防止求梯度时候，0导致的Nan
        # 这里不再乘以系数了
    else:
        # v2 版本的loss，已经不需要开方了
        loss2 = F.mse_loss(box_pred_response[:, 2:],
                           box_target_response[:, 2:],
                           reduction='none')
    loss2 = (s_wh * loss2).sum() / N

    # loss 3 & loss 4
    if not conf_bce:
        # loss 3 置信度误差(含有物体)  contain_loss
        loss3 = F.mse_loss(confidence_pred_response,
                           confidence_target_response, reduction='sum') / N

        # loss 4 置信度误差(不含目标)  not_contain_loss + no_obj_loss
        loss4 = F.mse_loss(confidence_pred_no_response,
                           confidence_target_no_response, reduction='sum') / N
    else:
        # v3 版本的loss
        bcel = nn.BCEWithLogitsLoss(reduction='sum')

        # loss 3 置信度误差(含有物体)  contain_loss
        loss3 = bcel(confidence_pred_response,
                     confidence_target_response) / N

        # loss 4 置信度误差(不含目标)  not_contain_loss + no_obj_loss
        loss4 = bcel(confidence_pred_no_response,
                     confidence_target_no_response) / N

    if not cls_ce:
        # loss 5 分类预测误差
        loss5 = F.mse_loss(class_pred_response,
                           class_target_response, reduction='sum') / N
    else:
        # v3 版本的loss
        ce = nn.CrossEntropyLoss(reduction='sum')
        loss5 = ce(class_pred_response, class_target_response) / N

    # loss 6 初始化误差(v2 版本引入的)
    # 开始的几个batch，进行初始化，使得预测的宽高接近锚框
    if global_step_num >= 0 and global_step_num < init_batch_num:
    # if (batch_cnt < init_batch_num) and (epoch == 0):
        xywh_target = torch.ones_like(boxes_pred)  # BS, S, S, B, 4
        xywh_target = xywh_target * torch.tensor([0.5, 0.5, 0, 0],
                                                 device=pred.device)[None, None, None, None, :]

        loss6 = F.mse_loss(boxes_pred, xywh_target, reduction='sum') / N
    else:
        loss6 = torch.tensor(0, device=pred.device)

    return torch.stack([loss1, loss2, loss3, loss4, loss5, loss6])

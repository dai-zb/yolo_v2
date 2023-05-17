# encoding:utf-8

import torch

class DetectionCollate(object):
    # 为了对detection_collate函数进行封装的(类似偏函数的功能)
    def __init__(self, img_reader, aug, encoder):
        self.img_reader = img_reader
        self.aug = aug
        self.encoder = encoder
    
    def __call__(self, batch):
        return detection_collate(batch,
                                 self.img_reader,
                                 self.aug,
                                 self.encoder) 


def detection_collate(batch, img_reader, aug, encoder):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    imgs = []
    targets =[]
    masks =[]
    img_names = []
    wh = []
    
    """
    sample[0]   图片的路径     file_path
    sample[1]   宽             w
    sample[2]   高             h
    sample[3]   边框(对角表示)  boxes_conner
    sample[4]   标签(编号)     labels_idx
    sample[5]   图片的名称     filename
    """

    for sample in batch:
        # sample 对应的就是Dateset对象__getitem__ 方法的返回
        img = img_reader(sample[0])
        
        w = sample[1]
        h = sample[2]
        wh.append([w, h])
        
        boxes_corner = torch.tensor(sample[3])

        classes = torch.tensor(sample[4])

        # 图像增广可能会改变图像的宽高 (w, h) 即为处理后
        # image_size为ToTensor()之后的图片宽高
        img, (w, h), boxes_corner, classes, image_size = aug(img, boxes_corner, classes)

        boxes_corner = boxes_corner / torch.tensor([w, h, w, h])[None, :]
        
        S = image_size // 32
        # 默认，图片尺寸下采样32，即为grid数量

        target, mask = encoder(boxes_corner, classes, S)

        imgs.append(img)
        targets.append(target)
        masks.append(mask)
        img_names.append(sample[5])

    # 返回 img_names  是为了根据名字加载图片和标注
    # 返回 wh         是因为在计算框的实际尺寸时用得到
    return torch.stack(imgs, 0), torch.stack(targets, 0), torch.stack(masks, 0), img_names, torch.tensor(wh)

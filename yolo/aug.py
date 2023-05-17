import random
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch

def aug(img, boxes_corner:torch.Tensor, labels:torch.Tensor,
        is_train=True,
        aug_lst=['random_flip','random_scale','random_blur',
                 'random_brightness','random_hue','random_saturation',
                'random_shift','random_crop'],
        image_size=448):
    if is_train:
        for fun_str in aug_lst:
            img, boxes_corner, labels = eval(fun_str)(img, boxes_corner, labels)

    h, w, _ = img.shape
    img = BGR2RGB(img) # because pytorch pretrained model use RGB

    img = cv2.resize(img, (image_size, image_size))

    img = transforms.ToTensor()(img)

    # (w, h)  是图片增广之后的宽高，返回是为了编码时，处理boxes使用
    # image_size  是因为之后要根据此判断grid cell的数量
    return img, (w, h), boxes_corner, labels, image_size


def BGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def BGR2HSV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def HSV2BGR(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def random_blur(bgr, boxes, labels,
                p=0.5,
                ksize:'核大小'=(5, 5)):
    """
    均值滤波
    """
    if random.random() < p:
        bgr = cv2.blur(bgr, ksize)
    return bgr, boxes, labels


def random_brightness(bgr, boxes, labels,
                      p=0.5,
                      r:""=[0.5, 1.5]):
    if random.random() < p:
        hsv = BGR2HSV(bgr)
        h,s,v = cv2.split(hsv)
        adjust = random.choice(r)
        v = v*adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h,s,v))
        bgr = HSV2BGR(hsv)
    return bgr, boxes, labels


def random_saturation(bgr, boxes, labels,
                      p=0.5,
                      r:""=[0.5, 1.5]):
    if random.random() < p:
        hsv = BGR2HSV(bgr)
        h,s,v = cv2.split(hsv)
        adjust = random.choice(r)
        s = s*adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h,s,v))
        bgr = HSV2BGR(hsv)
    return bgr, boxes, labels


def random_hue(bgr, boxes, labels,
               p=0.5,
               r:""=[0.5, 1.5]):
    if random.random() < p:
        hsv = BGR2HSV(bgr)
        h,s,v = cv2.split(hsv)
        adjust = random.choice(r)
        h = h*adjust
        h = np.clip(h, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h,s,v))
        bgr = HSV2BGR(hsv)
    return bgr, boxes, labels


def random_shift(bgr, boxes, labels,
                 p=0.5, r=0.2):
    # 平移变换
    # 会出现灰色边
    center = (boxes[:,2:]+boxes[:,:2])/2
    if random.random() < p:
        height, width, c = bgr.shape
        after_shfit_image = np.zeros((height,width,c), dtype=bgr.dtype)
        after_shfit_image[:, :, :] = (104, 117, 123) #bgr
        shift_x = random.uniform(-width*r, width*r)
        shift_y = random.uniform(-height*r, height*r)
        #print(bgr.shape,shift_x,shift_y)
        #原图像的平移
        if shift_x>=0 and shift_y>=0:
            after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]
        elif shift_x>=0 and shift_y<0:
            after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
        elif shift_x <0 and shift_y >=0:
            after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
        elif shift_x<0 and shift_y<0:
            after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]

        shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
        center = center + shift_xy
        mask1 = (center[:,0] >0) & (center[:,0] < width)
        mask2 = (center[:,1] >0) & (center[:,1] < height)
        mask = (mask1 & mask2).view(-1,1)
        boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
        if len(boxes_in) == 0:
            return bgr, boxes, labels
        box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)
        boxes_in = boxes_in+box_shift
        labels_in = labels[mask.view(-1)]
        return after_shfit_image, boxes_in, labels_in
    return bgr, boxes, labels


def random_scale(bgr, boxes:torch.Tensor, labels, 
                 p=0.5,
                 ratio:'缩放比例'=[0.8, 1.2]):
    # 只是resize
    # 固定住高度，以指定比例伸缩宽度，做图像形变
    # 感觉用处不大
    if random.random() < p:
        scale = random.uniform(ratio[0], ratio[1])
        height,width,c = bgr.shape
        bgr = cv2.resize(bgr,(int(width*scale),height))
        scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            
        boxes = boxes * scale_tensor
        return bgr, boxes, labels
    return bgr, boxes, labels


def random_crop(bgr, boxes:torch.Tensor, labels,
               p=0.5,
               ratio:'裁剪比例'=0.6):
    if random.random() < p:
        center = (boxes[:,2:]+boxes[:,:2])/2
        height,width,c = bgr.shape
        h = random.uniform(ratio*height,height)
        w = random.uniform(ratio*width,width)
        x = random.uniform(0,width-w)
        y = random.uniform(0,height-h)
        x,y,h,w = int(x),int(y),int(h),int(w)

        center = center - torch.FloatTensor([[x,y]]).expand_as(center)
        mask1 = (center[:,0]>0) & (center[:,0]<w)
        mask2 = (center[:,1]>0) & (center[:,1]<h)
        mask = (mask1 & mask2).view(-1,1)

        boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
        if(len(boxes_in)==0):
            return bgr, boxes, labels
        box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)

        boxes_in = boxes_in - box_shift
        boxes_in[:,0]=boxes_in[:,0].clamp_(min=0,max=w)
        boxes_in[:,2]=boxes_in[:,2].clamp_(min=0,max=w)
        boxes_in[:,1]=boxes_in[:,1].clamp_(min=0,max=h)
        boxes_in[:,3]=boxes_in[:,3].clamp_(min=0,max=h)

        labels_in = labels[mask.view(-1)]
        img_croped = bgr[y:y+h,x:x+w,:]
        return img_croped, boxes_in, labels_in
    return bgr, boxes, labels


def random_flip(im, boxes, labels,
                p=0.5):
    if random.random() < p:
        im_lr = np.fliplr(im).copy()
        h,w,_ = im.shape

        xmin = w - boxes[:,2]
        xmax = w - boxes[:,0]

        boxes[:,0] = xmin
        boxes[:,2] = xmax
        return im_lr, boxes, labels
    return im, boxes, labels

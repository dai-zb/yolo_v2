# coding: utf-8

import os
import xml.etree.ElementTree as ET

COLOR = [[0, 0, 0],
         [128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128],
         [128, 0, 128],
         [0, 128, 128],
         [128, 128, 128],
         [64, 0, 0],
         [192, 0, 0],
         [64, 128, 0],
         [192, 128, 0],
         [64, 0, 128],
         [192, 0, 128],
         [64, 128, 128],
         [192, 128, 128],
         [0, 64, 0],
         [128, 64, 0],
         [0, 192, 0],
         [128, 192, 0],
         [0, 64, 128]]

VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


an = '/home/aigeek/dataset/VOC%s/Annotations/'
im = '/home/aigeek/dataset/VOC%s/JPEGImages/'

def parse(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)

    boxes_conner = []
    labels = []
    labels_idx = []
    
    filename = tree.find('filename').text
    size = tree.find('size')
    
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    c = int(size.find('depth').text)

    for obj in tree.findall('object'):
        obj_struct = {}

        difficult = int(obj.find('difficult').text)
        if difficult == 1:  # difficult  0  表示易于识别
            # print(filename)
            continue

        cls_name = obj.find('name').text
        class_idx = VOC_CLASSES.index(cls_name)
        
        #obj_struct['pose'] = obj.find('pose').text
        #obj_struct['truncated'] = int(obj.find('truncated').text)
        #obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        bbox = [int(float(bbox.find('xmin').text)),
                int(float(bbox.find('ymin').text)),
                int(float(bbox.find('xmax').text)),
                int(float(bbox.find('ymax').text))]

        boxes_conner.append(bbox)
        labels_idx.append(class_idx)
        labels.append(cls_name)

    return { 'w':w, 'h':h, 'c':c,
            'filename': filename,
            'boxes_conner':boxes_conner,
            'labels_idx':labels_idx,
            'labels':labels}

"""
train_data = get_dataset(['2007', '2012'],
                         ['file_path', 'w', 'h', 'boxes_conner', 'labels_idx', 'filename'])
test_data = get_dataset(['2007-test'],
                         ['file_path', 'w', 'h', 'boxes_conner', 'labels_idx', 'filename'])
"""
def get_dataset(names, keys):
    if type(names) is not list:
        names = [names]
    annotations_path = [an % x for x in names]
    img_path = [im % x for x in names]
    return VocDataset(annotations_path, img_path, keys)


class VocDataset(object):
    """
    可以加载的key:
       w, h, c       宽 高 通道数
       filename      图片名称
       file_path     图片路径
       boxes_conner  框(对角表示)
       labels_idx    标签(数字编号)
       labels        标签(文本)
    会自动将'difficult'的框给筛选掉
            
    """
    def __init__(self, annotations_path, img_path, keys):
        assert type(annotations_path) is list
        assert type(img_path) is list
        assert len(annotations_path) == len(img_path)
        
        xml_files = []
        base_path = []
        
        for an_p, im_p in zip(annotations_path, img_path):
            lst = os.listdir(an_p)
            base_lst = [(an_p, im_p)]*len(lst)
            xml_files += lst
            base_path += base_lst
            print('read ' + str(len(lst)) + ' labels from  ' + an_p)

        self.keys = keys

        print('keys: ' + str(keys))
        
        # 解析，因为要过滤掉标记为空的图片(因为跳过了difficult的框)
        self.lst = []
        for (an_p, im_p), xml_file in zip(base_path, xml_files):
            # an_p  注解的路径
            # im_p  图像的路径
            d = parse(an_p + xml_file)
            d['file_path'] = im_p + d['filename']
            
            if len(d['labels']) != 0:
                self.lst.append(d)
            else:
                print('Empty label: '+d['filename'])

        self.len = len(self.lst)
        print('total not empty files: ' + str(self.len))

    def __getitem__(self, idx):
        if idx >= self.len:
            raise StopIteration()
            
        d = self.lst[idx]
        return tuple(d[x] for x in self.keys)


    def __len__(self):
        return self.len

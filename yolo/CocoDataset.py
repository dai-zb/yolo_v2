# coding: utf-8

import json
import random

an = '/root/jupyter_workhome/dataset/coco/coco%s/annotations/instances_%s%s.json'
im = '/root/jupyter_workhome/dataset/coco/coco%s/%s%s/'

with open('/root/jupyter_workhome/dataset/coco/coco91_to_80.json') as f:
    coco91_to_80_json = json.load(f)

# coco的id，是从1开始的
coco91_to_80_minus1 = {}
coco80_minus1_to_91 = {}

for k in coco91_to_80_json:
    # 从0开始计算，因为coco是从1开编号的
    coco91_to_80_minus1[int(k)] = coco91_to_80_json[k]-1
    coco80_minus1_to_91[coco91_to_80_json[k]-1] = int(k)

with open('/root/jupyter_workhome/dataset/coco/coco80_indices.json') as f:
    coco80_indices_json = json.load(f)

COCO_CLASSES = []

for k in range(1, 81):
    COCO_CLASSES.append(coco80_indices_json[str(k)])

    
voc20_to_coco80_minus1 = {}

with open('/root/jupyter_workhome/dataset/coco/voc20_to_coco80.json') as f:
    voc20_to_coco80_json = json.load(f)

for k in voc20_to_coco80_json:
    voc20_to_coco80_minus1[int(k)] = voc20_to_coco80_json[k]-1
    

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
         [0, 64, 128]]*5


def get_dataset(names, val_train, keys):
    """
    coco.get_dataset(2017, 'train', keys)
    coco.get_dataset(2017, 'val', keys)
    """
    if type(names) is not list:
        names = [names]
    if type(val_train) is not list:
        val_train = [val_train]
    annotations_path = [an %(n, v, n) for n,v in zip(names, val_train)]
    img_path = [im %(n, v, n) for n,v in zip(names, val_train)]
    return CocoDataset(annotations_path, img_path, keys)


def annotations_2_dict(annotations, recode):
    d = {}
    for an in annotations:
        image_id = an['image_id']
        bbox = an['bbox']
        category_id = an['category_id']
        if recode:
            category_id = coco91_to_80_minus1[category_id]
        
        dd = d.get(image_id, {'boxes_conner':[], 'category_id':[]})
        if bbox[2]==0 or bbox[3]==0:
            print('w or h 为0', an)
            continue
        dd['boxes_conner'].append([bbox[0],
                                   bbox[1],
                                   bbox[0] + bbox[2], 
                                   bbox[1] + bbox[3]])
        dd['category_id'].append(category_id)
        
        d[image_id] = dd
    return d


def random_select(lst, select_len):
    if select_len < 0:
        return lst
    
    if select_len >= len(lst):
        return lst
    
    random.shuffle(lst)
    return lst[:select_len]


class CocoDataset(object):
    """
    可以加载的key:
       width, height  宽 高
       filename       图片名称
       file_path      图片路径
       id             图片的id
       boxes_conner   框(对角表示)
       category_id    标签(数字编号)  0~79
    """
    def __init__(self, annotations_path, img_path, keys,
                 select_len=-1, recode=True):
        """
        annotations_path   coco的json
        img_path           图片所在的路径
        keys               要加载的项
        select_len         可以随机选取n个图片
        recode             是否将分类重新编码(到0~79)
        """
        assert type(annotations_path) is list
        assert type(img_path) is list
        assert len(annotations_path) == len(img_path)
        
        self.images = []
        
        for an_p, im_p in zip(annotations_path, img_path):
            with open(an_p) as f:
                coco = json.load(f)
            
            images = coco['images']
            annotations = coco['annotations']
            # 将读入的json转为指定格式的
            # 会将分类的编号转为 0~79
            an_d = annotations_2_dict(annotations, recode=recode)
            del annotations
            
            id_lst_no_bbox = [] 
            
            for k,img in enumerate(images):
                # 尝试删除用不到的内容，减少内存
                if 'license' in img:
                    del img['license']
                if 'coco_url' in img:
                    del img['coco_url']
                if 'date_captured' in img:
                    del img['date_captured']
                if 'flickr_url' in img:
                    del img['flickr_url']
                img['file_path'] = im_p + img['file_name']
                img_id = img['id']
                bc = an_d.get(img_id)
                if bc is None:
                    # print('no bbox id: ' + str(img_id))
                    id_lst_no_bbox.append(img_id)
                    continue
                img['boxes_conner'] = bc['boxes_conner']
                img['category_id'] = bc['category_id']
            
            images = list(filter(lambda x: x['id'] not in id_lst_no_bbox, images))
                
            print('read %d(%d) labels from %s' % (len(images), k+1, an_p))
            
            self.images += images
            del coco
            del an_d

        self.keys = keys
        
        self.images = random_select(self.images, select_len)
        
        self.len = len(self.images)
        print('keys: ' + str(keys))
        print('total image files: ' + str(self.len))

    def __getitem__(self, idx):
        if idx >= self.len:
            raise StopIteration()
        # if idx >= 13000:
        #     raise StopIteration()
            
        d = self.images[idx]
        return tuple(d[x] for x in self.keys)

    def __len__(self):
        # if self.len >= 13000:
        #     return 13000
        return self.len

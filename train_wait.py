#!/usr/bin/env python
# coding: utf-8

import sys

assert len(sys.argv)==3 or len(sys.argv)==2

if len(sys.argv)==3:
    cnt = int(sys.argv[1])
    assert cnt > 10 and cnt < 100
    
    cmd = 'python train.py ' + ' '.join(sys.argv[2:])
else:
    cnt = 15
    cmd = 'python train.py ' + ' '.join(sys.argv[1:])

print('cnt: ' + str(cnt))
print('cmd: ' + cmd)

# 等待资源

sys.path.append("/root/jupyter_workhome/my_packages/tricks/") 

import wait_resource as wait

wait.wait_gpu(_cnt=cnt)

# 运行训练程序

import os

os.system(cmd)

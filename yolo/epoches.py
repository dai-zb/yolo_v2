#!/usr/bin/env python
# coding: utf-8

import threading
import time
import pandas as pd
import monitor as mon


class MonitorThread(threading.Thread):
    def __init__(self, fun, table, delay):
        threading.Thread.__init__(self)
        self.fun = fun
        self.table = table
        self.delay = delay
        self.delay_10 = delay / 20
        self.flag = 1

    def run(self):
        while self.flag:
            d = self.fun()
            self.table.update(d)
            for _ in range(10):
                if self.flag == 0:
                    break
                time.sleep(self.delay_10)  # 这样可以减少延时等待
        self.flag = -1

    def stop(self):
        self.flag = 0
        while self.flag != -1:
            time.sleep(0.1)

            
class Table(object):
    def __init__(self, file_path):
        if file_path == 'console':
            self.print_to = 'console'
        elif file_path is None:
            self.print_to = 'None'
        else:
            self.print_to = 'csv'
            self.file_path = file_path
            try:
                self.df = pd.read_csv(file_path)
                print(file_path + ' already exists')
            except FileNotFoundError:
                self.df = pd.DataFrame()
                print(file_path + ' will be created')

            try:
                self.df.set_index('id', inplace=True)
            except KeyError:
                pass

        
    def update(self, d):
        if self.print_to == 'console':
            print(d)
        elif self.print_to == 'None':
            pass
        else:
            self.df = self.df.append(d, ignore_index=True)
            self.df.index.name = 'id'
            self.df.to_csv(self.file_path)
            # self.df.to_csv(self.file_path, float_format='%.4f')

"""
注意，传入的函数func，必须要接收两个参数，分别表示当前的epoch和总共的epoch
也必须返回一个字典类型的对象，用于记录

否则会出错

def train(epoch, epoch_num):
   pass
   return {"train_loss":0.5}
"""
def epoches(epoch_num, func,
            func_table_path='./func.csv',
            monitor=mon.cpu_mem_gpu_monitor,
            monitor_delay=20,
            mon_table_path='./monitor.csv'):
    
    func_table = Table(func_table_path)
    
    # 某些情况下，可以关闭对服务器资源占用的监控
    should_monitor = (monitor is not None) and (monitor_delay is not None) and (mon_table_path is not None)
    
    if should_monitor:
        mon_th = MonitorThread(monitor, Table(mon_table_path), monitor_delay)
        mon_th.start()
    
    lst = []
    
    for epoch_id in range(epoch_num):
        ts_start = time.time()
        
        # 调用核心方法
        ret_dict = func(epoch_id, epoch_num)
        
        ts_end =time.time()
        
        d = {"ts_start": ts_start, 
             "ts_end": ts_end,
             "epoch_id": epoch_id,
             "epoch_num":epoch_num}
        
        if ret_dict is not None:
            d.update(ret_dict)
            func_table.update(d)
            lst.append(d)
        
    if should_monitor:
        mon_th.stop()
    
    return pd.DataFrame(lst)


"""
注意，传入的函数func，必须要接收两个参数，分别表示当前的epoch和总共的epoch
也必须返回一个字典类型的对象，用于记录

否则会出错

def train(epoch, epoch_num):
   pass
   return {"train_loss":0.5}
"""
    
# 装饰器，可以更简便使用
def epoches_n(epoch_num,
              func_table_path='./func.csv',
              monitor=mon.cpu_mem_gpu_monitor,
              monitor_delay=20,
              mon_table_path='./monitor.csv'):
    print('train epoches num', epoch_num)
    def decrator(func):
        def fun():
            return epoches(epoch_num, func,
                      func_table_path=func_table_path,
                      monitor=monitor,
                      monitor_delay=monitor_delay,
                      mon_table_path=mon_table_path)
        return fun
    return decrator
    
# 装饰器，可以更简便使用
def epoches_n_try(epoch_num,
              func_table_path='./func.csv',
              monitor=mon.cpu_mem_gpu_monitor,
              monitor_delay=20,
              mon_table_path='./monitor.csv'):
    print('train epoches num', epoch_num)
    def decrator(func):
        def fun():
            try:
                return epoches(epoch_num, func,
                          func_table_path=func_table_path,
                          monitor=monitor,
                          monitor_delay=monitor_delay,
                          mon_table_path=mon_table_path)
            except KeyboardInterrupt:
                print('被终止')
        return fun
    return decrator
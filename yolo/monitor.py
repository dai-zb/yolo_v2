#!/usr/bin/env python
# coding: utf-8

import os


# {'gpu_power_used': 24, 'gpu_power_total': 75, 'gpu_mem_used': 2073, 'gpu_mem_total': 7611, 'gpu_used': 0}
def nvidia_smi(raw_info=False):
    gpu_info = os.popen('nvidia-smi').read()

    lst = gpu_info.split('\n')[9]
    lst = lst.split('|')

    power = lst[1].split()
    power_used = power[3][:-1]
    power_total = power[5][:-1]

    mem = lst[2].split()
    mem_used = mem[0][:-3]
    mem_total = mem[2][:-3]

    gpu_used = lst[3].split()[0][:-1]

    d = {'gpu_power_used': int(power_used),
         'gpu_power_total': int(power_total),
         'gpu_mem_used': int(mem_used),
         'gpu_mem_total': int(mem_total),
         'gpu_used': int(gpu_used)}
    if raw_info:
        d['gpu_info'] = gpu_info

    return d


# {'mem_used': 2745, 'mem_total': 15885}
def free(raw_info=False):
    mem_info = os.popen('free -mw').read()
    # total = used + free + buffers + cache
    # free：未被分配的内存。
    # buffers：系统分配但未被使用的buffers数量
    # cached：系统分配但未被使用的cache数量
    # available = free + buffers + cache  # 不知道为什么，感觉free返回的available计算不准
    lst = mem_info.split('\n')[1].split()

    total = lst[1]
    used = lst[2]

    d = {'mem_used': int(used), 'mem_total': int(total)}
    if raw_info:
        d['free_info'] = mem_info

    return d


# {'cpu_used': 1.5999999999999943, 'mem_used': 2745, 'mem_total': 15885}
def top(raw_info=False):
    top_info = os.popen('top -bn 1 -p 0').read()
    # us：用户空间占用CPU百分比
    # sy：内核空间占用CPU百分比
    # ni：(nice)用户进程空间内改变过优先级的进程占用CPU百分比
    # id：(idle)空闲CPU百分比
    # wa：(wait)等待输入输出的CPU时间百分比
    # hi：硬件中断
    # si：软件中断
    # st：实时

    lst = top_info.split('\n')

    cpu_info = lst[2]
    cpu_info = cpu_info.split(':')[1]
    cpu_used = cpu_info.split(',')[3]
    cpu_used = cpu_used.split()[0]
    cpu_used = 100 - float(cpu_used)

    mem_info = lst[3]
    mem_info = mem_info.split(':')[1]
    mem_info = mem_info.split(',')

    mem_used = mem_info[2]
    mem_used = mem_used.split()[0]
    mem_used = int(mem_used) // 1024

    mem_total = mem_info[0]
    mem_total = mem_total.split()[0]
    mem_total = int(mem_total) // 1024

    d = {'cpu_used': cpu_used, 'mem_used': mem_used, 'mem_total': mem_total}
    if raw_info:
        d['top_info'] = top_info

    return d


import time

def _try(func):
    def f():
        try:
            return func()
        except BaseException:
            print('获取监控数据异常')
            return {'monitor_ts': time.time()}

    return f


@_try
def cpu_mem_gpu_monitor():
    d = {'monitor_ts': time.time()}
    d.update(top())
    d.update(nvidia_smi())
    return d


@_try
def cpu_mem_monitor():
    d = {'monitor_ts': time.time()}
    d.update(top())
    return d

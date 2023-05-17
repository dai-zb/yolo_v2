from collections import defaultdict

def append_dict_dict_list(key1, key2, item, d=None):
    if d is None:
        d = defaultdict(lambda: defaultdict(list))
    d[key1][key2].append(item)
    return d

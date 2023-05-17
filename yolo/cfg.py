import json


def _update_dict(a, b):
    assert isinstance(a, dict)
    assert isinstance(b, dict)
    for k in b:
        if isinstance(b[k], dict) and a.get(k) is not None:
            _update_dict(a[k], b[k])
        else:
            a[k] = b[k]


def read_json(paths):
    assert isinstance(paths, list)
    lst = []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            lst.append(json.load(f))
    return lst


class Cfg(object):

    def __init__(self, d):
        if not isinstance(d, list):
            assert isinstance(d, dict)
            self.dict = d
        else:
            self.dict = {}
            for dd in d:
                assert isinstance(dd, dict)
                _update_dict(self.dict, dd)

    def __getattr__(self, item):
        if isinstance(self.dict[item], dict):
            return Cfg(self.dict[item])
        return self.dict.get(item, None)
    
    def __getitem__(self, item):
        if '(comment)' in item:
            return None
        return self.dict[item]

    def __setitem__(self, key, value):
        if '(comment)' in key:
            return None
        self.dict[key] = value

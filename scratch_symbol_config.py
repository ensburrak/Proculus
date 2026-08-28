import json

with open('config.json') as f:
    c = json.load(f)

def find_keys(obj, key):
    res = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key: res.append(v)
            res.extend(find_keys(v, key))
    elif isinstance(obj, list):
        for item in obj:
            res.extend(find_keys(item, key))
    return res

print('max_symbols_per_loop:', find_keys(c, 'max_symbols_per_loop'))
print('max_symbols_per_cycle:', find_keys(c, 'max_symbols_per_cycle'))
print('always_on_symbols:', find_keys(c, 'always_on_symbols'))
print('breadth_symbols limit:', find_keys(c, 'breadth_max_symbols'))

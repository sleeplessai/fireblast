import logging

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)


def _check_anns(name, anns):
  logging.warning(f'Checking {name} annotation existence')
  for k, v in anns.items():
    if not v.exists():
      anns[k] = None
      logging.warning(f'{name}.{k} not exists.')


def _ann_to_list(ann_file, varts_idx):
  ann_str_list = [l.rstrip('\n') for l in open(ann_file, 'r').readlines()]
  sample_list = []
  for s in ann_str_list:
    t = s.find(' ')
    vi = [s[:t], varts_idx[s[t + 1:]]]
    sample_list.append(vi)
  return sample_list


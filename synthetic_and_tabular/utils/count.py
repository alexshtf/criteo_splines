#!/usr/bin/env python3

import argparse, csv, sys, collections

from common import *
from tqdm import tqdm

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
args = vars(parser.parse_args())


pos_counts = collections.Counter()
neg_counts = collections.Counter()
fields = [f'C{j}' for j in range(1, 27)]

for i, row in tqdm(enumerate(csv.DictReader(open(args['csv_path'])), start=1), mininterval=0.25):
    label = row['Label']
    keys = [field + ',' + row[field] for field in fields]
    if label == '0':
        neg_counts.update(keys)
    else:
        pos_counts.update(keys)
total_counts = pos_counts + neg_counts

print('Field,Value,Neg,Pos,Total,Ratio')
for key, total in sorted(total_counts.items(), key=lambda pair: pair[1]):
    if total < 10:
        continue

    pos = pos_counts[key]
    neg = neg_counts[key]

    ratio = round(float(pos)/total, 5)
    print(key+','+str(neg)+','+str(pos)+','+str(total)+','+str(ratio))

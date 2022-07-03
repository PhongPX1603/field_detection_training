import os
import sys
sys.path.append(os.environ['PWD'])
import json
import argparse

from pathlib import Path
from natsort import natsorted
from collections import defaultdict
from core.handlers.logger import Logger

def verbose(message):
    print(message)
    logger.info(msg=message)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-dir', type=str)
    parser.add_argument('--mask-pattern', type=str)
    args = parser.parse_args()

    logger = Logger(save_dir=f'core/datasets/data_distribution/{str(Path(args.label_dir).name)}').get_logger(log_name='data_distribute')

    label_paths = natsorted(Path(args.label_dir).glob('**/*{}'.format(args.mask_pattern)), key=lambda x: x.stem)
    
    verbose(message=f'This folder have: {len(label_paths)} files')

    fields = defaultdict(int)
    for label_path in label_paths:
        with open(label_path) as fp:
            info = json.load(fp)['shapes']
        for field_info in info:
            if not field_info['label'].startswith('K'):
                fields[field_info['label']] += 1
    
    fields = sorted(fields.items(), key =lambda kv:(kv[1], kv[0]))
    for field, distribute in fields.items():
        verbose(message=f'{field}: {distribute}')
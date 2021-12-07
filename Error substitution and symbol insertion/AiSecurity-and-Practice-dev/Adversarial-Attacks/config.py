import csv
import json
from collections import namedtuple

import torch

IMG_SIZE = 299
USE_CUDA = False
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

TRUE_LABEL_CSV_PATH = './supports/LOC_val_solution.csv'

IDX2SYN_LABEL = json.load(open('./supports/imagenet_class_index.json'))

DATASET_IMG_PATH = '/path/to/CLS-LOC-DATASET/'

IMG_NAME2SYN = {}
for i in csv.reader(open(TRUE_LABEL_CSV_PATH, 'r')):
    if i[0].split('_')[0] == 'ILSVRC2012':
        IMG_NAME2SYN[i[0]] = i[1][0:9]

PredLabel = namedtuple('PredLabel', ['img_name', 'top1_idx', 'top1_syn', 'top1_label', 'top1_prob',
                                     'top5_idx', 'top5_syn', 'top5_label', 'top5_prob'])

# img_info: ImgInfo, others: PredLabel
PredLabels = namedtuple('PredLabels', ['img_info', 'clean', 'FGSM', 'BIM', 'LL'])

ImgInfo = namedtuple('ImgInfo', ['clean', 'FGSM', 'BIM', 'LL', 'syn'])

TEMPLATE_CONFIG = {
    'PAGE_WIDTH': int(8.27 * 300),
    'PAGE_HEIGHT': int(11.7 * 300),  # 2481 * 3510
    'QR_SIZE': 300,
    'IMG_SIZE': 1000,
    'QR_MARGIN': 100,
    'IMG_MARGIN_X': 200,
    'IMG_MARGIN_Y': 710
}

SELECTED_EPS = (8, 16)

CONF = TEMPLATE_CONFIG
MODEL_QR_POS = {
    'tl': (CONF['QR_MARGIN'] + CONF['QR_SIZE'] / 2,
           CONF['QR_MARGIN'] + CONF['QR_SIZE'] / 2),
    'tr': (CONF['PAGE_WIDTH'] - CONF['QR_MARGIN'] - CONF['QR_SIZE'] / 2,
           CONF['QR_MARGIN'] + CONF['QR_SIZE'] / 2),
    'bl': (CONF['QR_MARGIN'] + CONF['QR_SIZE'] / 2,
           CONF['PAGE_HEIGHT'] - CONF['QR_MARGIN'] - CONF['QR_SIZE'] / 2),
    'br': (CONF['PAGE_WIDTH'] - CONF['QR_MARGIN'] - CONF['QR_SIZE'] / 2,
           CONF['PAGE_HEIGHT'] - CONF['QR_MARGIN'] - CONF['QR_SIZE'] / 2)
}

IMG_POS = {
    'clean': ((CONF['IMG_MARGIN_X'], CONF['IMG_MARGIN_Y']),
              (CONF['IMG_MARGIN_X'] + CONF['IMG_SIZE'], CONF['IMG_MARGIN_Y'] + CONF['IMG_SIZE'])),
    'FGSM': ((CONF['PAGE_WIDTH'] - CONF['IMG_MARGIN_X'] - CONF['IMG_SIZE'], CONF['IMG_MARGIN_Y']),
             (CONF['PAGE_WIDTH'] - CONF['IMG_MARGIN_X'], CONF['IMG_MARGIN_Y'] + CONF['IMG_SIZE'])),
    'BIM': ((CONF['IMG_MARGIN_X'], CONF['PAGE_HEIGHT'] - CONF['IMG_MARGIN_Y'] - CONF['IMG_SIZE']),
            (CONF['IMG_MARGIN_X'] + CONF['IMG_SIZE'], CONF['PAGE_HEIGHT'] - CONF['IMG_MARGIN_Y'])),
    'LL': ((CONF['PAGE_WIDTH'] - CONF['IMG_MARGIN_X'] - CONF['IMG_SIZE'],
            CONF['PAGE_HEIGHT'] - CONF['IMG_MARGIN_Y'] - CONF['IMG_SIZE']),
           (CONF['PAGE_WIDTH'] - CONF['IMG_MARGIN_X'], CONF['PAGE_HEIGHT'] - CONF['IMG_MARGIN_Y']))
}

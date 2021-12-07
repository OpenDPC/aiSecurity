import os
import pickle
from typing import List, Union, Tuple, Any, Dict, Optional

import cv2
import numpy as np
import qrcode
from PIL import Image
from matplotlib import pyplot as plt
from pyzbar import pyzbar

from config import *


# * Filter and loader
def name_filter(name: str) -> bool:
    return name.split('.')[-1] in ['JPG', 'JPEG', 'PNG', 'jpg', 'jpeg', 'png']


def load_util_img(img_path: str, req_tensor: bool = True) -> Union[torch.Tensor, np.ndarray]:
    assert os.path.isfile(img_path), f'Image "{img_path}" does not exist.'
    img = cv2.imread(img_path)
    width, height = img.shape[:2]
    if width != height:
        side_len = min(width, height)
        if width > height:
            start_pixel = round((width - side_len) / 2) + 1
            img = img[start_pixel:start_pixel + side_len, :, :]
        else:
            start_pixel = round((height - side_len) / 2) + 1
            img = img[:, start_pixel:start_pixel + side_len, :]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.tensor(np.transpose(img / 255, (2, 0, 1)), dtype=torch.float) if req_tensor else img


def load_pickles(*, file_path: str = None, base_path: str = None) -> List[Any]:
    assert file_path is not None or base_path is not None, 'Must use either file_path or base_path.'
    if file_path is not None:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    file_ls = os.listdir(base_path)
    file_ls.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    res_ls = []
    for file in file_ls:
        with open(os.path.join(base_path, file), 'rb') as f:
            res_ls += pickle.load(f)
    return res_ls


# * Transformer
def idx2syn_label(x: torch.Tensor) -> Tuple[Union[str, List[str]], ...]:
    x = rm_tensor(x)
    if isinstance(x, int):
        return IDX2SYN_LABEL[str(x)][0], IDX2SYN_LABEL[str(x)][1]
    elif isinstance(x, list):
        syn_temp, label_temp = [], []
        for i_ in x:
            syn_temp.append(IDX2SYN_LABEL[str(i_)][0]), label_temp.append(IDX2SYN_LABEL[str(i_)][1])
        return syn_temp, label_temp
    else:
        raise TypeError('Input must be in type torch.Tensor.')


def rm_tensor(x: torch.Tensor) -> Union[int, List[int]]:
    res = x.numpy().tolist()
    return res[0] if len(res) == 1 else res


def name2syn(img_name: str) -> str:
    assert img_name.split('.')[-1] in ['JPG', 'JPEG', 'PNG', 'jpg', 'jpeg', 'png'], 'img_name error.'
    if len(img_name.split('_')) > 3:
        img_name = '_'.join(img_name.split('_')[-3:])
    return IMG_NAME2SYN[img_name.split('.')[0]]


# * Photo transformation
def create_template(info: ImgInfo, save_base_path: str, config: Dict[str, Any] = TEMPLATE_CONFIG):
    page = Image.new('RGB', (config['PAGE_WIDTH'], config['PAGE_HEIGHT']), 'white')
    for pos in ['tl', 'tr', 'bl', 'br']:
        img = qrcode.make(f'{pos}:{info.syn}', border=0).resize((config['QR_SIZE'], config['QR_SIZE']))
        if pos == 'tl':
            page.paste(img, box=(config['QR_MARGIN'], config['QR_MARGIN']))
        elif pos == 'tr':
            page.paste(img, box=(config['PAGE_WIDTH'] - config['QR_MARGIN'] - config['QR_SIZE'], config['QR_MARGIN']))
        elif pos == 'bl':
            page.paste(img, box=(config['QR_MARGIN'], config['PAGE_HEIGHT'] - config['QR_MARGIN'] - config['QR_SIZE']))
        else:
            page.paste(img, box=(config['PAGE_WIDTH'] - config['QR_MARGIN'] - config['QR_SIZE'],
                                 config['PAGE_HEIGHT'] - config['QR_MARGIN'] - config['QR_SIZE']))
    # Clean image
    img = Image.fromarray(load_util_img(info.clean, False)).resize((config['IMG_SIZE'], config['IMG_SIZE']))
    page.paste(img, box=(config['IMG_MARGIN_X'], config['IMG_MARGIN_Y']))
    # FGSM image
    img = Image.fromarray(load_util_img(info.FGSM, False)).resize((config['IMG_SIZE'], config['IMG_SIZE']))
    page.paste(img, box=(config['PAGE_WIDTH'] - config['IMG_MARGIN_X'] - config['IMG_SIZE'], config['IMG_MARGIN_Y']))
    # BIM image
    img = Image.fromarray(load_util_img(info.BIM, False)).resize((config['IMG_SIZE'], config['IMG_SIZE']))
    page.paste(img, box=(config['IMG_MARGIN_X'], config['PAGE_HEIGHT'] - config['IMG_MARGIN_Y'] - config['IMG_SIZE']))
    # LL image
    img = Image.fromarray(load_util_img(info.LL, False)).resize((config['IMG_SIZE'], config['IMG_SIZE']))
    page.paste(img, box=(config['PAGE_WIDTH'] - config['IMG_MARGIN_X'] - config['IMG_SIZE'],
                         config['PAGE_HEIGHT'] - config['IMG_MARGIN_Y'] - config['IMG_SIZE']))
    page.save(os.path.join(save_base_path, f'temp_{os.path.basename(info.clean).split(".")[0]}.png'))


def photo_extractor(img_path: str) -> Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    assert os.path.isfile(img_path), f'Image "{img_path}" does not exist.'
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    barcodes = pyzbar.decode(gray)
    img_qr_pos = {}
    syn = ''
    for barcode in barcodes:
        data = barcode.data.decode('utf-8')
        pos_info, syn = data.split(':')
        (x, y, w, h) = barcode.rect
        center_pos = (x + w / 2, y + h / 2)
        img_qr_pos[pos_info] = center_pos
    model_anchors = np.float32([MODEL_QR_POS['tl'], MODEL_QR_POS['tr'], MODEL_QR_POS['br'], MODEL_QR_POS['bl']])
    img_anchors = np.float32([img_qr_pos['tl'], img_qr_pos['tr'], img_qr_pos['br'], img_qr_pos['bl']])
    m = cv2.getPerspectiveTransform(img_anchors, model_anchors)
    dst = cv2.warpPerspective(img, m, (TEMPLATE_CONFIG['PAGE_WIDTH'], TEMPLATE_CONFIG['PAGE_HEIGHT']))
    dst_1 = dst[IMG_POS['clean'][0][1]:IMG_POS['clean'][1][1], IMG_POS['clean'][0][0]:IMG_POS['clean'][1][0], :]
    dst_2 = dst[IMG_POS['FGSM'][0][1]:IMG_POS['FGSM'][1][1], IMG_POS['FGSM'][0][0]:IMG_POS['FGSM'][1][0], :]
    dst_3 = dst[IMG_POS['BIM'][0][1]:IMG_POS['BIM'][1][1], IMG_POS['BIM'][0][0]:IMG_POS['BIM'][1][0], :]
    dst_4 = dst[IMG_POS['LL'][0][1]:IMG_POS['LL'][1][1], IMG_POS['LL'][0][0]:IMG_POS['LL'][1][0], :]
    return syn, dst_1, dst_2, dst_3, dst_4


# * Error calculation
def cal_acc_single(pred_res: PredLabel, real_syn: str = None) -> Tuple[bool, bool]:
    if real_syn is None:
        real_syn = name2syn(pred_res.img_name)
    return real_syn == pred_res.top1_syn, real_syn in pred_res.top5_syn


# * Visualization
def acc_dis_chart_plt(y1, y2, y3, y4, _mode, img_name):
    x = np.array([2, 8, 16, 24, 32, 40, 48])
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, y1, color='blue', label="clean img")
    ax.plot(x, y2, color='green', label="FGSM", marker='o')
    ax.plot(x, y3, color='red', label="BIM", marker='v')
    ax.plot(x, y4, color='purple', label="LL", marker='s')

    ax.set_xlabel('epsilon')
    if _mode == "top1":
        ax.set_ylabel('Top-1 Accuracy')
    else:
        ax.set_ylabel('Top-5 Accuracy')

    ax.grid(True, linestyle='-.')
    ax.set_xticks([2, 8, 16, 24, 32, 40, 48])
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1.15))

    plt.title(" ")
    plt.xlabel("epsilon")
    plt.savefig(os.path.join('./results', img_name) + '.png')

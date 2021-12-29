import os.path
from typing import Any, Optional

import cv2
import numpy as np
from torchattacks import FGSM, BIM
from torchvision import models

from config import *
from utils import load_util_img, idx2syn_label, rm_tensor
from torchvision import utils as v_utils


class Normalize(torch.nn.Module):
    # Define a normalization layer
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input_):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input_ - mean) / std


# Define and load Inception-v3
inception_v3 = models.inception_v3(init_weights=False)
pre_param = torch.load('./supports/inception_v3_google-0cc3c7bd.pth')
inception_v3.load_state_dict(pre_param)
# Adding a normalization layer for inception-v3.
MODEL = torch.nn.Sequential(
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    inception_v3,
).to(DEVICE).eval()  # eval() changes the model to evaluation state (lock parameters).


def predict(img_path: str) -> PredLabel:
    img = load_util_img(img_path).to(DEVICE).unsqueeze(0)
    with torch.no_grad():
        output = MODEL(img)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # The output has un-normalized scores. To get probabilities, run a softmax on it.
    top1_prob, top1_cat_id = torch.topk(probabilities, 1)
    top5_prob, top5_cat_id = torch.topk(probabilities, 5)
    return PredLabel(os.path.basename(img_path),
                     rm_tensor(top1_cat_id), *idx2syn_label(top1_cat_id), rm_tensor(top1_prob),
                     rm_tensor(top5_cat_id), *idx2syn_label(top5_cat_id), rm_tensor(top5_prob))


def adv_img_gen(img_path: str, atk_method: str, eps: float, save_pre_path: str = None,
                *, to_file: bool = True, color_type: str = 'RGB',
                model: Any = MODEL, alpha=2 / 255, steps=100, label=0) -> Optional[np.ndarray]:
    if os.path.isfile(os.path.join(save_pre_path, f'{atk_method}_{eps}_{os.path.basename(img_path)}')):
        return
    assert color_type in ['RGB', 'BGR'], 'color_type must be RGB or BGR.'
    if to_file:
        assert os.path.exists(save_pre_path), 'save_pre_path does not exist.'
    img = load_util_img(img_path).to(DEVICE).unsqueeze(0)
    eps_ = eps / 255
    label = torch.LongTensor((torch.FloatTensor([label])).numpy())
    if atk_method == 'FGSM':
        atk = FGSM(model, eps_)
    elif atk_method == 'BIM':
        atk = BIM(model, eps_, alpha, steps)
    elif atk_method == 'LL':
        atk = BIM(model, eps_, alpha, steps)
        atk.set_mode_targeted_least_likely(1)
    else:
        raise ValueError('atk_method must in ("FGSM", "BIM", "LL").')
    atk.set_return_type(type='float')
    if USE_CUDA:
        adv_img = atk(img, label)
        if not to_file:
            return adv_img
        v_utils.save_image(adv_img, os.path.join(save_pre_path, f'{atk_method}_{eps}_{os.path.basename(img_path)}'),
                           normalize=True)
    else:
        adv_img = np.transpose(atk(img, label).numpy()[0, :, :, :], (1, 2, 0))
        if not to_file:
            return adv_img
        adv_img = cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_pre_path, f'{atk_method}_{eps}_{os.path.basename(img_path)}'), adv_img * 255)

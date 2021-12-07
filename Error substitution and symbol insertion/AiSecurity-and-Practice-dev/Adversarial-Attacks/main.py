"""
@author: Yuze Xuan, Runnan Zhu, Xuan Wang, Ziyi Mo, Lehan Kang
"""
import random
from math import ceil
from multiprocessing import Pool

from model import predict, adv_img_gen
from utils import *

if __name__ == '__main__':
    # * Predict all clean images
    img_name_ls = os.listdir(os.path.join(DATASET_IMG_PATH, 'clean'))
    img_name_ls = list(filter(name_filter, img_name_ls))
    img_path_ls = [os.path.join(DATASET_IMG_PATH, 'clean', img_name) for img_name in img_name_ls]
    chunks_size = 5000
    chunks_num = ceil(len(img_path_ls) // chunks_size)
    if not os.path.exists('./results/clean_pred'):
        os.makedirs('./results/clean_pred')
    if USE_CUDA:  # ! GPU method
        # TODO: Needs test
        for idx in range(chunks_num):
            if idx == chunks_num - 1:
                pred_res_ls = list(map(predict, img_path_ls[idx * chunks_size:]))
            else:
                pred_res_ls = list(map(predict, img_path_ls[idx * chunks_size:(idx + 1) * chunks_size]))
            with open(f'./results/clean_pred/pred_res_ls_{idx}.pickle', 'wb') as f:
                pickle.dump(pred_res_ls, f)
            print(f'Prediction progress: {idx + 1} of {chunks_num}')
    else:  # ! CPU Pool method
        pool = Pool()
        for idx in range(chunks_num):
            if idx == chunks_num - 1:
                pred_res_ls = pool.map(predict, img_path_ls[idx * chunks_size:])
            else:
                pred_res_ls = pool.map(predict, img_path_ls[idx * chunks_size:(idx + 1) * chunks_size])
            with open(f'./results/clean_pred/pred_res_ls_{idx}.pickle', 'wb') as f:
                pickle.dump(pred_res_ls, f)
            print(f'Prediction progress: {idx + 1} of {chunks_num}')
    # * Calculate accuracy for the prediction of all clean images
    pred_res_ls: List[PredLabel] = load_pickles(base_path='./results/clean_pred')
    top1_acc_ls, top5_acc_ls = [], []
    for res in pred_res_ls:
        top1_temp, top5_temp = cal_acc_single(res)
        top1_acc_ls.append(top1_temp)
        top5_acc_ls.append(top5_temp)
    print('Accuracy on clean images:\n',
          f'Top-1: {round(sum(top1_acc_ls) / len(top1_acc_ls), 4)}\n',
          f'Top-5: {round(sum(top5_acc_ls) / len(top5_acc_ls), 4)}')
    # * Choose 50 correctly-predicted-images based on Top-1 accuracy, prob > 0.6
    random.seed(2)
    random.shuffle(pred_res_ls)
    selected_imgs: List[PredLabel] = []
    for res in pred_res_ls:
        if cal_acc_single(res)[0] and res.top1_prob > 0.6:
            selected_imgs.append(res)
            if len(selected_imgs) == 50:
                break
    with open(f'./results/selected_images.pickle', 'wb') as f:
        pickle.dump(selected_imgs, f)
    # * Generate adversarial images, epsilon in (2, 8, 16, 24, 32, 40, 48)
    selected_imgs: List[PredLabel] = load_pickles(file_path='./results/selected_images.pickle')
    for method in ['BIM']:
        for eps in (32, 40, 48):
            print('Generating adversarial samples with method:', method, ', epsilon =', eps)
            if not os.path.exists(os.path.join(DATASET_IMG_PATH, method, str(eps))):
                os.makedirs(os.path.join(DATASET_IMG_PATH, method, str(eps)))
            if USE_CUDA:  # ! GPU method
                for sample in selected_imgs:
                    adv_img_gen(os.path.join(DATASET_IMG_PATH, 'clean', sample.img_name),
                                method, eps, os.path.join(DATASET_IMG_PATH, method, str(eps)))
            else:  # ! CPU Pool method
                pool = Pool(processes=10)
                for sample in selected_imgs:
                    pool.apply_async(adv_img_gen, (os.path.join(DATASET_IMG_PATH, 'clean', sample.img_name),
                                                   method, eps, os.path.join(DATASET_IMG_PATH, method, str(eps))))
                pool.close()
                pool.join()
    # * Predict selected images (clean and adversarial)
    selected_imgs: List[PredLabel] = load_pickles(file_path='./results/selected_images.pickle')
    for method in ['clean', 'FGSM', 'BIM', 'LL']:
        if not os.path.exists(f'./results/selected_{method}_pred/'):
            os.makedirs(f'./results/selected_{method}_pred/')
        if method == 'clean':
            selected_img_path_ls = [os.path.join(DATASET_IMG_PATH, 'clean', item.img_name) for item in
                                    selected_imgs]
            if USE_CUDA:  # ! GPU method
                # TODO: Needs test
                pred_selected_res_ls = list(map(predict, selected_img_path_ls))
            else:  # ! CPU Pool method
                pool = Pool()
                pred_selected_res_ls = pool.map(predict, selected_img_path_ls)
            with open(f'./results/selected_{method}_pred/{method}_res_ls.pickle', 'wb') as f:
                pickle.dump(pred_selected_res_ls, f)
            print(f'Prediction progress: Type = {method}')
        else:
            for eps in (2, 8, 16, 24, 32, 40, 48):
                selected_img_path_ls = [os.path.join(DATASET_IMG_PATH, method, str(eps),
                                                     f'{method}_{eps}_{item.img_name}') for item in selected_imgs]
                if USE_CUDA:  # ! GPU method
                    # TODO: Needs test
                    pred_selected_res_ls = list(map(predict, selected_img_path_ls))
                else:  # ! CPU Pool method
                    pool = Pool()
                    pred_selected_res_ls = pool.map(predict, selected_img_path_ls)
                with open(f'./results/selected_{method}_pred/{method}_{eps}_res_ls.pickle', 'wb') as f:
                    pickle.dump(pred_selected_res_ls, f)
                print(f'Prediction progress: Type = {method}, Epsilon = {eps}')
    # * Calculate accuracy for the prediction of selected images
    top1_acc_dict, top5_acc_dict = {}, {}
    for method in ['clean', 'FGSM', 'BIM', 'LL']:
        if method == 'clean':
            pred_selected_res_ls: List[PredLabel] = load_pickles(
                file_path=f'./results/selected_{method}_pred/{method}_res_ls.pickle')
            top1_acc_ls, top5_acc_ls = [], []
            for res in pred_selected_res_ls:
                top1_temp, top5_temp = cal_acc_single(res)
                top1_acc_ls.append(top1_temp)
                top5_acc_ls.append(top5_temp)
            # print('Accuracy on selected clean images:\n',
            #       f'Top-1: {round(sum(top1_acc_ls) / len(top1_acc_ls), 4)}\n',
            #       f'Top-5: {round(sum(top5_acc_ls) / len(top5_acc_ls), 4)}')
            top1_acc_dict[method] = [round(sum(top1_acc_ls) / len(top1_acc_ls), 4)] * 7
            top5_acc_dict[method] = [round(sum(top5_acc_ls) / len(top5_acc_ls), 4)] * 7
        else:
            top1_acc_dict[method], top5_acc_dict[method] = [], []
            for eps in (2, 8, 16, 24, 32, 40, 48):
                pred_selected_res_ls: List[PredLabel] = load_pickles(
                    file_path=f'./results/selected_{method}_pred/{method}_{eps}_res_ls.pickle')
                top1_acc_ls, top5_acc_ls = [], []
                for res in pred_selected_res_ls:
                    top1_temp, top5_temp = cal_acc_single(res)
                    top1_acc_ls.append(top1_temp)
                    top5_acc_ls.append(top5_temp)
                # print(f'Accuracy on selected images modified by {method}, epsilon = {eps}:\n',
                #       f'Top-1: {round(sum(top1_acc_ls) / len(top1_acc_ls), 4)}\n',
                #       f'Top-5: {round(sum(top5_acc_ls) / len(top5_acc_ls), 4)}')
                top1_acc_dict[method].append(round(sum(top1_acc_ls) / len(top1_acc_ls), 4))
                top5_acc_dict[method].append(round(sum(top5_acc_ls) / len(top5_acc_ls), 4))
    acc_dis_chart_plt(top1_acc_dict['clean'], top1_acc_dict['FGSM'], top1_acc_dict['BIM'], top1_acc_dict['LL'],
                      'top1', 'selected-top1')
    acc_dis_chart_plt(top5_acc_dict['clean'], top5_acc_dict['FGSM'], top5_acc_dict['BIM'], top5_acc_dict['LL'],
                      'top5', 'selected-top5')
    # * Generate templates
    selected_imgs: List[PredLabel] = load_pickles(file_path='./results/selected_images.pickle')
    for eps in SELECTED_EPS:
        if not os.path.exists(os.path.join(DATASET_IMG_PATH, 'template', str(eps))):
            os.makedirs(os.path.join(DATASET_IMG_PATH, 'template', str(eps)))
        for img in selected_imgs:
            create_template(ImgInfo(os.path.join(DATASET_IMG_PATH, 'clean', img.img_name),
                                    os.path.join(DATASET_IMG_PATH, 'FGSM', str(eps), f'FGSM_{eps}_{img.img_name}'),
                                    os.path.join(DATASET_IMG_PATH, 'BIM', str(eps), f'BIM_{eps}_{img.img_name}'),
                                    os.path.join(DATASET_IMG_PATH, 'LL', str(eps), f'LL_{eps}_{img.img_name}'),
                                    name2syn(img.img_name)), os.path.join(DATASET_IMG_PATH, 'template', str(eps)))
    # * Extract photos
    # TODO: Modify names
    for eps in SELECTED_EPS:
        file_ls = os.listdir(os.path.join(DATASET_IMG_PATH, f'photo/{eps}'))
        file_ls = list(filter(name_filter, file_ls))
        for file in file_ls:
            syn, img1, img2, img3, img4 = photo_extractor(
                os.path.join(DATASET_IMG_PATH, f'photo/{eps}', file))
            if not os.path.exists(os.path.join(DATASET_IMG_PATH, f'extracted_photo/{eps}', syn)):
                os.makedirs(os.path.join(DATASET_IMG_PATH, f'extracted_photo/{eps}', syn))
            cv2.imwrite(os.path.join(DATASET_IMG_PATH, f'extracted_photo/{eps}', syn, 'clean.png'), img1)
            cv2.imwrite(os.path.join(DATASET_IMG_PATH, f'extracted_photo/{eps}', syn, 'FGSM.png'), img2)
            cv2.imwrite(os.path.join(DATASET_IMG_PATH, f'extracted_photo/{eps}', syn, 'BIM.png'), img3)
            cv2.imwrite(os.path.join(DATASET_IMG_PATH, f'extracted_photo/{eps}', syn, 'LL.png'), img4)
    # * Predict extracted photos
    top1_acc_dict, top5_acc_dict = {}, {}
    for method in ['clean', 'FGSM', 'BIM', 'LL']:
        top1_acc_dict[method], top5_acc_dict[method] = {}, {}
        for eps in SELECTED_EPS:
            top1_acc_dict[method][eps], top5_acc_dict[method][eps] = [], []
            file_ls = os.listdir(os.path.join(DATASET_IMG_PATH, 'extracted_photo', str(eps)))
            file_ls = [i for i in file_ls if i[0] == 'n']
            for file in file_ls:
                pred_res = predict(os.path.join(DATASET_IMG_PATH, 'extracted_photo', str(eps), file, f'{method}.png'))
                top1_acc_dict[method][eps].append(cal_acc_single(pred_res, file)[0])
                top5_acc_dict[method][eps].append(cal_acc_single(pred_res, file)[1])
            print(f'Current method: {method}, epsilon: {eps}, top1:',
                  round(sum(top1_acc_dict[method][eps]) / len(top1_acc_dict[method][eps]), 4),
                  'top5:', round(sum(top5_acc_dict[method][eps]) / len(top5_acc_dict[method][eps]), 4))

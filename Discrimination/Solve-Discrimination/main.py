#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on 2021/10/19
@author: Yuze Xuan, Runnan Zhu, Xuan Wang, Ziyi Mo, Lehan Kang
"""
import os
import pickle
from collections import namedtuple, Counter
from multiprocessing.pool import Pool
from typing import Tuple, Optional, Union, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Dataset = namedtuple('Dataset', ('data', 'label', 'protect'))

# Generating random seed sequence
np.random.seed(1)
SEEDS = list(np.random.randint(1, 1000, 50000))


def get_seed() -> int: return SEEDS.pop(0)


def randomize(dataset: Dataset) -> Dataset:
    """ Randomly shuffle the dataset.
    :param dataset: Dataset Object.
    :return: A Dataset Object after randomize.
    """
    _dataset = np.hstack((dataset.data, dataset.label.reshape((-1, 1)), dataset.protect.reshape((-1, 1))))
    np.random.seed(get_seed())
    np.random.shuffle(_dataset)
    return Dataset(_dataset[:, :-2], _dataset[:, -2].ravel(), _dataset[:, -1].ravel())


def mode(x: np.ndarray) -> np.ndarray:
    """ Mode by line.
    :param x: np.ndarray Object.
    :return: A 1-d np.ndarray.
    """
    _res = []
    for idx in range(x.shape[0]):
        count_dict = Counter(x[idx, :])
        _label = 0 if count_dict[0] > count_dict[1] else 1
        _res.append(_label)
    return np.array(_res)


def data_utils(names_file_path: str, data_file_path: str, positive_label: Union[int, str], protect_feature_name: str,
               split_rate: float, *, protect_feature_values: Optional[Tuple[str, ...]] = None,
               protect_feature_condition: Optional[Tuple[str, Union[int, float]]] = None,
               data_sep: str = ',', header: bool = False, use_filter: bool = True, use_only_numeric_label: bool = False,
               filter_ignore_tup: Tuple[str, str] = ('continuous', 'numeric')) -> Dict[str, Dataset]:
    """ Load and pre-treat the dataset with custom requirements.
    Basic Attribute
    :param names_file_path: The path of the file includes features name and set of alternative labels.
    :param data_file_path: The path of the data file.
    :param positive_label: Positive sample label which corresponding to 1 in numeric label.
    :param protect_feature_name: The name of the protected feature.
    :param split_rate: Train and test dataset split rate.
    Optional Attribute
    :param protect_feature_values: Protected sample tag value list.
    :param protect_feature_condition: Protected sample condition. A tuple of condition + base value. eg: (>, 65)
    :param data_sep: The separation character of the data if not comma.
    :param header: Whether the data file includes header.
    :param use_filter: Whether to use the value filter.
    :param use_only_numeric_label: Use only numeric features and delete label features.
    :param filter_ignore_tup: The ignore tuple for the filter.
    :returns: Datasets include train, test and unlabeled in a dictionary.
    """
    # Assert check
    assert protect_feature_values is not None or protect_feature_condition is not None, \
        'Either protect_attr_value or protect_attr_condition should be specified.'
    if protect_feature_condition is not None:
        assert protect_feature_condition[0] in ('<', '>', '<=', '>='), \
            'Protect_attr_condition only supports operation of (<, >, <=, >=).'
    # Pre-define the container variable
    info_dict = {}
    names_ls = []
    # Load names file
    with open(names_file_path) as f:
        names = f.readlines()
    # Extract names
    for item in names:
        item = item.rstrip().replace(' ', '')
        name, labels = item.split(':')
        names_ls.append(name)
        info_dict[name] = labels.split(',')
    if names_ls[-1] != 'label':
        names_ls.append('label')
    # Load data file
    names_ls = None if header else names_ls
    _data = pd.read_csv(data_file_path, sep=data_sep, names=names_ls, index_col=False)
    # Data filter
    if use_filter:
        for name in _data.columns:
            try:
                if info_dict[name][0] not in filter_ignore_tup:
                    _data = _data[_data[name].isin(info_dict[name])]
            except KeyError:
                del _data[name]
    # Get labels
    _label = np.zeros((_data.shape[0], 1), np.uint8)
    _label[_data['label'] == positive_label] = 1
    _label = _label.ravel()
    del _data['label']
    # Classify the protection group
    _protect = np.zeros((_data.shape[0], 1), np.uint8)
    if protect_feature_values is not None:
        _protect[_data[protect_feature_name].isin(protect_feature_values)] = 1
    else:
        try:
            base_value = float(protect_feature_condition[1])
        except ValueError:
            raise ValueError('The second value of protect_attr_condition should be in type int or float.')
        exec('_protect[_data[protect_feature_name]' + protect_feature_condition[0] + 'base_value] = 1')
    del _data[protect_feature_name]
    _protect = _protect.ravel()
    # Encode data with OneHot
    if use_only_numeric_label:
        for name in _data.columns:
            if info_dict[name][0] not in filter_ignore_tup:
                del _data[name]
    else:
        _data = pd.get_dummies(_data)
    _data = np.array(_data)
    # Split dataset
    dataset = randomize(Dataset(_data, _label, _protect))
    sample_num = len(dataset.label)
    labeled_num = round(sample_num / 2)
    train_num = round(labeled_num * split_rate)
    return {'train': Dataset(dataset.data[:train_num, :], dataset.label[:train_num], dataset.protect[:train_num]),
            'test': Dataset(dataset.data[train_num:labeled_num, :], dataset.label[train_num:labeled_num],
                            dataset.protect[train_num:labeled_num]),
            'unlabeled': Dataset(dataset.data[labeled_num:, :], dataset.label[labeled_num:],
                                 dataset.protect[labeled_num:])}


def cal_discrimination(pred_label: np.ndarray, _protect: np.ndarray) -> float:
    """ Calculate the discrimination of the predicted label. """
    protect_num = sum(_protect == 1)
    unprotect_num = sum(_protect == 0)
    return abs(sum((pred_label == 1) & (_protect == 1)) / protect_num - sum(
        (pred_label == 1) & (_protect == 0)) / unprotect_num)


def fairness_sampling(dataset: Dict[str, Dataset], model_name: str, _rho: float, _k: int) -> np.ndarray:
    """ Fairness-enhanced Sampling framework.
    :param dataset: Datasets include train, test and unlabeled in a dictionary.
    :param model_name: lr or svm.
    :param _rho: Sampling ratio.
    :param _k: Ensemble size.
    :return: Array of predict label.
    """
    # Where to sample
    assert model_name in ['lr', 'svm'], 'Model only supports operation lr and svm.'
    # Sampling based on rho
    _unlabeled = randomize(dataset['unlabeled'])
    unlabeled_num = round(len(_unlabeled.label) * _rho)
    # Train a model with original training dataset to predict the unlabeled data
    _model = LogisticRegression(max_iter=5000) if model_name == 'lr' else svm.SVC()
    _model.fit(dataset['train'].data, dataset['train'].label)
    # Predict the unlabeled data
    _unlabeled_data = _unlabeled.data[:unlabeled_num, :]
    _unlabeled_pred_label = _model.predict(_unlabeled_data)
    # print(accuracy_score(_unlabeled.label[:unlabeled_num], _unlabeled_pred_label))
    _new_dataset = Dataset(np.vstack((dataset['train'].data, _unlabeled_data)),
                           np.hstack((dataset['train'].label, _unlabeled_pred_label)).ravel(),
                           np.hstack((dataset['train'].protect, _unlabeled.protect[:unlabeled_num])).ravel())
    # How to sample & How to train the model
    _new_dataset_sep = {}
    sample_num_ls = []
    # Divide into groups
    for _label_idx in range(2):
        for _protect_idx in range(2):
            _index = (_new_dataset.label == _label_idx) & (_new_dataset.protect == _protect_idx)
            _new_dataset_sep[str(_label_idx) + str(_protect_idx)] = Dataset(_new_dataset.data[_index, :],
                                                                            _new_dataset.label[_index],
                                                                            _new_dataset.protect[_index])
            sample_num_ls.append(sum(_index))
    ns = sorted(sample_num_ls)[1]  # Take the second smallest sample number as ns
    # Resampling
    _pred_result = None
    for i in range(_k):
        print('curr iter:', i)
        _data = None
        _label = None
        _protect = None
        for _label_idx in range(2):
            for _protect_idx in range(2):
                _curr_group = randomize(_new_dataset_sep[str(_label_idx) + str(_protect_idx)])
                sample_num = len(_curr_group.label)
                if sample_num < ns:
                    _times = ns // sample_num + 1
                    _curr_group = randomize(Dataset(np.repeat(_curr_group.data, _times, axis=0),
                                                    np.repeat(_curr_group.label, _times),
                                                    np.repeat(_curr_group.protect, _times)))
                _data = _curr_group.data[:ns, :] if _data is None else np.vstack((_curr_group.data[:ns, :], _data))
                _label = _curr_group.label[:ns] if _label is None else np.hstack((_curr_group.label[:ns], _label))
                _protect = _curr_group.protect[:ns] if _protect is None else np.hstack(
                    (_curr_group.protect[:ns], _protect))
        fair_dataset = randomize(Dataset(_data, _label, _protect))
        _model = LogisticRegression(max_iter=5000) if model_name == 'lr' else svm.SVC()
        _model.fit(fair_dataset.data, fair_dataset.label)
        _curr_res = _model.predict(dataset['test'].data).reshape((-1, 1))
        _pred_result = _curr_res if _pred_result is None else np.hstack((_pred_result, _curr_res))
    return mode(_pred_result)


def test_rho(_info) -> Tuple[float, float]:
    dataset: Dict[str, Dataset] = _info['dataset']
    model_name: str = _info['model_name']
    _rho: float = _info['rho']
    _pred_label = fairness_sampling(dataset, model_name, _rho, 200)
    return accuracy_score(dataset['test'].label, _pred_label), cal_discrimination(_pred_label, dataset['test'].protect)


def test_k(_info) -> Tuple[float, float]:
    dataset: Dict[str, Dataset] = _info['dataset']
    model_name: str = _info['model_name']
    _k: int = _info['k']
    _pred_label = fairness_sampling(dataset, model_name, 1, _k)
    return accuracy_score(dataset['test'].label, _pred_label), cal_discrimination(_pred_label, dataset['test'].protect)


def trans_acc_dis(_res_ls) -> Tuple[np.ndarray, np.ndarray]:
    _acc = []
    _dis = []
    for _res in _res_ls:
        _acc.append(_res[0])
        _dis.append(_res[1])
    return np.array(_acc), np.array(_dis)


def acc_dis_chart_plt(y1, y2, _mode, img_name):
    x = np.arange(0, 1, 0.1) if _mode == 'rho' else np.array([1, 10, 50, 100, 200, 500])
    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax2 = ax1.twinx()
    ax1.plot(x, y1, color='red', label="Accuracy", marker='.')
    ax2.plot(x, y2, label="Discrimination", marker='d')
    ax1.set_xlabel('ρ')
    ax1.set_ylabel('Accuracy')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right', bbox_to_anchor=(1.1, 1.15))

    plt.title(" ")
    if _mode == 'rho':
        plt.xlabel("ρ")
    else:
        plt.xlabel("k")
    plt.ylabel("Discrimination")

    plt.savefig(os.path.join('Solve-Discrimination', 'res_img', img_name) + '.png')
    # plt.show()


def acc_dis_bar_plt(num_list1, num_list2, img_name):
    label_list = ['ORI', 'FS']
    x = range(len(num_list1))

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()
    rects1 = ax1.bar(x=x, height=num_list1, width=0.4, alpha=0.8, color='red', label="Accuracy")
    rects2 = ax2.bar(x=[i + 0.4 for i in x], height=num_list2, width=0.4, color='b', label="Discrimination")
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Discrimination")

    plt.xticks([index + 0.2 for index in x], label_list)
    plt.xlabel(" ")
    plt.title(" ")

    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right', bbox_to_anchor=(1.1, 1.15))

    plt.savefig(os.path.join('Solve-Discrimination', 'res_img', img_name) + '.png')
    # plt.show()


def save_data(data, _file_name):
    with open(os.path.join('Solve-Discrimination', 'cache', _file_name + '.pickle'), 'wb') as f:
        pickle.dump(data, f)


def load_data(_file_name):
    if _file_name.split('.')[-1] != 'pickle':
        _file_name = _file_name + '.pickle'
    with open(os.path.join('Solve-Discrimination', 'cache', _file_name), 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    pool = Pool()
    if not os.path.exists(os.path.join('Solve-Discrimination', 'res_img')):
        os.makedirs(os.path.join('Solve-Discrimination', 'res_img'))
    if not os.path.exists(os.path.join('Solve-Discrimination', 'cache')):
        os.makedirs(os.path.join('Solve-Discrimination', 'cache'))

    # Dataset Adult
    print('Loading adult.')
    adult = data_utils('Solve-Discrimination/data/adult/adult-names.txt',
                       'Solve-Discrimination/data/adult/adult.data', '>50K', 'sex', 0.8,
                       protect_feature_values=('Female',), header=True)

    # Changes of accuracy and discrimination with rho (k = 200)
    print('Calculating changes of accuracy and discrimination with rho on adult.')
    info = []
    for rho in np.arange(0.1, 1.1, 0.1):
        info.append({'dataset': adult, 'model_name': 'lr', 'rho': rho})
    adult_lr_rho_res_ls = pool.map(test_rho, info)
    save_data(adult_lr_rho_res_ls, 'adult_lr_rho')
    acc_dis_chart_plt(*trans_acc_dis(adult_lr_rho_res_ls), 'rho', 'adult_lr_rho')
    info = []
    for rho in np.arange(0.1, 1.1, 0.1):
        info.append({'dataset': adult, 'model_name': 'svm', 'rho': rho})
    adult_svm_rho_res_ls = pool.map(test_rho, info)
    save_data(adult_svm_rho_res_ls, 'adult_svm_rho')
    acc_dis_chart_plt(*trans_acc_dis(adult_svm_rho_res_ls), 'rho', 'adult_svm_rho')

    # Changes of accuracy and discrimination with k (rho = 1)
    print('Calculating changes of accuracy and discrimination with k on adult.')
    info = []
    for k in [1, 10, 50, 100, 200, 500]:
        info.append({'dataset': adult, 'model_name': 'lr', 'k': k})
    adult_lr_k_res_ls = pool.map(test_k, info)
    save_data(adult_lr_k_res_ls, 'adult_lr_k')
    acc_dis_chart_plt(*trans_acc_dis(adult_lr_k_res_ls), 'k', 'adult_lr_k')
    info = []
    for k in [1, 10, 50, 100, 200, 500]:
        info.append({'dataset': adult, 'model_name': 'svm', 'k': k})
    adult_svm_k_res_ls = pool.map(test_k, info)
    save_data(adult_svm_k_res_ls, 'adult_svm_k')
    acc_dis_chart_plt(*trans_acc_dis(adult_svm_k_res_ls), 'k', 'adult_svm_k')

    # Accuracy and Discrimination with original and fs framework (rho = 1, k = 200)
    print('Calculating changes of accuracy and discrimination with original and fs framework on adult.')
    model_lr = LogisticRegression(max_iter=5000)
    model_lr.fit(adult['train'].data, adult['train'].label)
    adult_lr_ori_res = model_lr.predict(adult['test'].data)
    adult_lr_ori_acc = accuracy_score(adult['test'].label, adult_lr_ori_res)
    adult_lr_ori_dis = cal_discrimination(adult_lr_ori_res, adult['test'].protect)

    adult_lr_fs_res = fairness_sampling(adult, 'lr', 1, 200)
    adult_lr_fs_acc = accuracy_score(adult['test'].label, adult_lr_fs_res)
    adult_lr_fs_dis = cal_discrimination(adult_lr_fs_res, adult['test'].protect)

    acc_dis_bar_plt([adult_lr_ori_acc, adult_lr_fs_acc], [adult_lr_ori_dis, adult_lr_fs_dis], 'adult_ori_fs_lr')

    model_svm = svm.SVC()
    model_svm.fit(adult['train'].data, adult['train'].label)
    adult_svm_ori_res = model_svm.predict(adult['test'].data)
    adult_svm_ori_acc = accuracy_score(adult['test'].label, adult_svm_ori_res)
    adult_svm_ori_dis = cal_discrimination(adult_svm_ori_res, adult['test'].protect)

    adult_svm_fs_res = fairness_sampling(adult, 'lr', 1, 200)
    adult_svm_fs_acc = accuracy_score(adult['test'].label, adult_svm_fs_res)
    adult_svm_fs_dis = cal_discrimination(adult_svm_fs_res, adult['test'].protect)

    acc_dis_bar_plt([adult_svm_ori_acc, adult_svm_fs_acc], [adult_svm_ori_dis, adult_svm_fs_dis], 'adult_ori_fs_svm')

    # Dataset Bank
    print('Loading bank.')
    bank = data_utils('Solve-Discrimination/data/bank/bank_names.txt',
                      'Solve-Discrimination/data/bank/bank-additional/bank-additional-full.csv', 'yes', 'age', 0.8,
                      protect_feature_condition=('>=', 60), data_sep=';', header=True)

    # Changes of accuracy and discrimination with rho (k = 200)
    print('Calculating changes of accuracy and discrimination with rho on bank.')
    info = []
    for rho in np.arange(0.1, 1.1, 0.1):
        info.append({'dataset': bank, 'model_name': 'lr', 'rho': rho})
    bank_lr_rho_res_ls = pool.map(test_rho, info)
    acc_dis_chart_plt(*trans_acc_dis(bank_lr_rho_res_ls), 'rho', 'bank_lr_rho')
    info = []
    for rho in np.arange(0.1, 1.1, 0.1):
        info.append({'dataset': bank, 'model_name': 'svm', 'rho': rho})
    bank_svm_rho_res_ls = pool.map(test_rho, info)
    acc_dis_chart_plt(*trans_acc_dis(bank_svm_rho_res_ls), 'rho', 'bank_svm_rho')

    # Changes of accuracy and discrimination with k (rho = 1)
    print('Calculating changes of accuracy and discrimination with k on bank.')
    info = []
    for k in [1, 10, 50, 100, 200, 500]:
        info.append({'dataset': bank, 'model_name': 'lr', 'k': k})
    bank_lr_k_res_ls = pool.map(test_k, info)
    acc_dis_chart_plt(*trans_acc_dis(bank_lr_k_res_ls), 'k', 'bank_lr_k')
    info = []
    for k in [1, 10, 50, 100, 200, 500]:
        info.append({'dataset': bank, 'model_name': 'svm', 'k': k})
    bank_svm_k_res_ls = pool.map(test_k, info)
    acc_dis_chart_plt(*trans_acc_dis(bank_svm_k_res_ls), 'k', 'bank_svm_k')

    # Accuracy and Discrimination with original and fs framework (rho = 1, k = 200)
    print('Calculating changes of accuracy and discrimination with original and fs framework on bank.')
    model_lr = LogisticRegression(max_iter=5000)
    model_lr.fit(bank['train'].data, bank['train'].label)
    bank_lr_ori_res = model_lr.predict(bank['test'].data)
    bank_lr_ori_acc = accuracy_score(bank['test'].label, bank_lr_ori_res)
    bank_lr_ori_dis = cal_discrimination(bank_lr_ori_res, bank['test'].protect)

    bank_lr_fs_res = fairness_sampling(bank, 'lr', 1, 200)
    bank_lr_fs_acc = accuracy_score(bank['test'].label, bank_lr_fs_res)
    bank_lr_fs_dis = cal_discrimination(bank_lr_fs_res, bank['test'].protect)

    acc_dis_bar_plt([bank_lr_ori_acc, bank_lr_fs_acc], [bank_lr_ori_dis, bank_lr_fs_dis], 'bank_ori_fs_lr')

    model_svm = svm.SVC()
    model_svm.fit(bank['train'].data, bank['train'].label)
    bank_svm_ori_res = model_svm.predict(bank['test'].data)
    bank_svm_ori_acc = accuracy_score(bank['test'].label, bank_svm_ori_res)
    bank_svm_ori_dis = cal_discrimination(bank_svm_ori_res, bank['test'].protect)

    bank_svm_fs_res = fairness_sampling(bank, 'lr', 1, 200)
    bank_svm_fs_acc = accuracy_score(bank['test'].label, bank_svm_fs_res)
    bank_svm_fs_dis = cal_discrimination(bank_svm_fs_res, bank['test'].protect)

    acc_dis_bar_plt([bank_svm_ori_acc, bank_svm_fs_acc], [bank_svm_ori_dis, bank_svm_fs_dis], 'bank_ori_fs_svm')

    # Dataset Health
    print('Loading health.')
    health = data_utils('Solve-Discrimination/data/health/health_names.txt',
                        'Solve-Discrimination/data/health/health_v1.csv', 1, 'Age', 0.8,
                        protect_feature_values=('65', '75', '85'), header=True)

    # Changes of accuracy and discrimination with rho (k = 200)
    print('Calculating changes of accuracy and discrimination with rho on health.')
    info = []
    for rho in np.arange(0.1, 1.1, 0.1):
        info.append({'dataset': health, 'model_name': 'lr', 'rho': rho})
    health_lr_rho_res_ls = pool.map(test_rho, info)
    acc_dis_chart_plt(*trans_acc_dis(health_lr_rho_res_ls), 'rho', 'health_lr_rho')
    info = []
    for rho in np.arange(0.1, 1.1, 0.1):
        info.append({'dataset': health, 'model_name': 'svm', 'rho': rho})
    health_svm_rho_res_ls = pool.map(test_rho, info)
    acc_dis_chart_plt(*trans_acc_dis(health_svm_rho_res_ls), 'rho', 'health_svm_rho')

    # Changes of accuracy and discrimination with k (rho = 1)
    print('Calculating changes of accuracy and discrimination with k on health.')
    info = []
    for k in [1, 10, 50, 100, 200, 500]:
        info.append({'dataset': health, 'model_name': 'lr', 'k': k})
    health_lr_k_res_ls = pool.map(test_k, info)
    acc_dis_chart_plt(*trans_acc_dis(health_lr_k_res_ls), 'k', 'health_lr_k')
    info = []
    for k in [1, 10, 50, 100, 200, 500]:
        info.append({'dataset': health, 'model_name': 'svm', 'k': k})
    health_svm_k_res_ls = pool.map(test_k, info)
    acc_dis_chart_plt(*trans_acc_dis(health_svm_k_res_ls), 'k', 'health_svm_k')

    # Accuracy and Discrimination with original and fs framework (rho = 1, k = 200)
    print('Calculating changes of accuracy and discrimination with original and fs framework on health.')
    model_lr = LogisticRegression(max_iter=5000)
    model_lr.fit(health['train'].data, health['train'].label)
    health_lr_ori_res = model_lr.predict(health['test'].data)
    health_lr_ori_acc = accuracy_score(health['test'].label, health_lr_ori_res)
    health_lr_ori_dis = cal_discrimination(health_lr_ori_res, health['test'].protect)

    health_lr_fs_res = fairness_sampling(health, 'lr', 1, 200)
    health_lr_fs_acc = accuracy_score(health['test'].label, health_lr_fs_res)
    health_lr_fs_dis = cal_discrimination(health_lr_fs_res, health['test'].protect)

    acc_dis_bar_plt([health_lr_ori_acc, health_lr_fs_acc], [health_lr_ori_dis, health_lr_fs_dis], 'health_ori_fs_lr')

    model_svm = svm.SVC()
    model_svm.fit(health['train'].data, health['train'].label)
    health_svm_ori_res = model_svm.predict(health['test'].data)
    health_svm_ori_acc = accuracy_score(health['test'].label, health_svm_ori_res)
    health_svm_ori_dis = cal_discrimination(health_svm_ori_res, health['test'].protect)

    health_svm_fs_res = fairness_sampling(health, 'lr', 1, 200)
    health_svm_fs_acc = accuracy_score(health['test'].label, health_svm_fs_res)
    health_svm_fs_dis = cal_discrimination(health_svm_fs_res, health['test'].protect)

    acc_dis_bar_plt([health_svm_ori_acc, health_svm_fs_acc], [health_svm_ori_dis, health_svm_fs_dis],
                    'health_ori_fs_svm')

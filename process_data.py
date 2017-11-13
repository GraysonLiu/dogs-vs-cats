import os
import numpy as np
import cv2
from tqdm import tqdm
from random import shuffle


def get_img_label(img_file_name):
    word_label = img_file_name.split('.')[0]
    if word_label == 'cat':
        return [1, 0]
    elif word_label == 'dog':
        return [0, 1]


def process_train_set_data(img_size, raw_data_dir, save_path):
    train_set_data = []
    for img_file_name in tqdm(os.listdir(raw_data_dir)):
        label = get_img_label(img_file_name)
        path = os.path.join(raw_data_dir, img_file_name)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        train_set_data.append([np.array(img), np.array(label)])
    shuffle(train_set_data)
    np.save(save_path, train_set_data)
    return train_set_data


def process_test_set_data(img_size, raw_data_dir, save_path):
    test_set_data = []
    for img_file_name in tqdm(os.listdir(raw_data_dir)):
        path = os.path.join(raw_data_dir, img_file_name)
        img_id = img_file_name.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        test_set_data.append([np.array(img), img_id])
    np.save(save_path, test_set_data)
    return test_set_data

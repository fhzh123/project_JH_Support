import os
import numpy as np
import pandas as pd
from datasets import load_dataset

def data_split_index(data_len: int, split_ratio: float = 0.1):

    split_num = int(data_len * split_ratio)

    split_index = np.random.choice(data_len, split_num, replace=False)
    origin_index = list(set(range(data_len)) - set(split_index))

    return origin_index, split_index

def sampling_index(data_len: int, split_ratio: float = 0.1):

    split_num = int(data_len * split_ratio)

    split_index = np.random.choice(data_len, split_num, replace=False)

    return split_index

def data_load(data_path:str = None, data_name:str = None, augmentation:str = None, sampling_ratio:float = None):

    total_src_list, total_trg_list = dict(), dict()

    if data_name == 'SST2':

        data_path = os.path.join(data_path,'SST2')

        # 1) Train data load
        train_dat = pd.read_csv(f'/nas_homes/dataset/text_classification/{data_name}/train.csv', names=['label', 'text'])
        sampling_ix = sampling_index(data_len=len(train_dat), split_ratio=sampling_ratio)
        total_src_list['train'] = train_dat['text'].tolist()
        total_src_list['train'] = list(map(total_src_list['train'].__getitem__, sampling_ix))
        total_trg_list['train'] = train_dat['label'].tolist()
        total_trg_list['train'] = list(map(total_trg_list['train'].__getitem__, sampling_ix))

        if augmentation is not None:
            train_aug_dat = pd.read_csv(f'train_SST2_{augmentation}.csv', names=['aug_text', 'label'])
            total_src_list['train'] = total_src_list['train'] + list(map(train_aug_dat['label'].tolist().__getitem__, sampling_ix))
            total_trg_list['train'] = total_trg_list['train'] + list(map(train_aug_dat['label'].tolist().__getitem__, sampling_ix))

        # 2) Valid data load
        test_dat = pd.read_csv(f'/nas_homes/dataset/text_classification/{data_name}/train.csv', names=['label', 'text'])
        total_src_list['valid'] = test_dat['text'].tolist()
        total_trg_list['valid'] = test_dat['label'].tolist()

        # 3) Test data load
        # test_dat = pd.read_csv(f'/nas_homes/dataset/text_classification/{data_name}/train.csv', names=['label', 'text'])
        total_src_list['test'] = test_dat['text'].tolist()
        total_trg_list['test'] = test_dat['label'].tolist()

    if data_name == 'Yelp_Full':

        data_path = os.path.join(data_path,'Yelp_Full')

        # 1) Train data load
        train_dat = pd.read_csv(f'/nas_homes/dataset/text_classification/{data_name}/train.csv', names=['label', 'text'])
        sampling_ix = sampling_index(data_len=len(train_dat), split_ratio=sampling_ratio)
        total_src_list['train'] = train_dat['text'].tolist()
        total_src_list['train'] = list(map(total_src_list['train'].__getitem__, sampling_ix))
        total_trg_list['train'] = train_dat['label'].tolist()
        total_trg_list['train'] = list(map(total_trg_list['train'].__getitem__, sampling_ix))

        if augmentation is not None:
            train_aug_dat = pd.read_csv(f'train_Yelp_Full_{augmentation}.csv', names=['aug_text', 'label'])
            total_src_list['train'] = total_src_list['train'] + list(map(train_aug_dat['aug_text'].tolist().__getitem__, sampling_ix))
            total_trg_list['train'] = total_trg_list['train'] + list(map(train_aug_dat['label'].tolist().__getitem__, sampling_ix))

        # 2) Valid data load
        test_dat = pd.read_csv(f'/nas_homes/dataset/text_classification/{data_name}/train.csv', names=['label', 'text'])
        total_src_list['valid'] = test_dat['text'].tolist()
        total_trg_list['valid'] = test_dat['label'].tolist()

        # 3) Test data load
        # test_dat = pd.read_csv(f'/nas_homes/dataset/text_classification/{data_name}/train.csv', names=['label', 'text'])
        total_src_list['test'] = test_dat['text'].tolist()
        total_trg_list['test'] = test_dat['label'].tolist()

    if data_name == 'IMDB':

        dataset = load_dataset("imdb")

        # origin_index, split_index = data_split_index(data_len=len(dataset['test']['text']), split_ratio=0.5)
        sampling_ix = sampling_index(data_len=len(dataset['train']['text']), split_ratio=sampling_ratio)

        # 1) Train data load
        total_src_list['train'] = np.array(dataset['train']['text']).tolist()
        total_src_list['train'] = list(map(total_src_list['train'].__getitem__, sampling_ix))
        total_trg_list['train'] = np.array(dataset['train']['label']).tolist()
        total_trg_list['train'] = list(map(total_trg_list['train'].__getitem__, sampling_ix))
        if augmentation is not None:
            train_aug_dat = pd.read_csv(f'train_IMDB_{augmentation}.csv')
            train_aug_dat['label'] = train_aug_dat['label'].replace('positive', 0)
            train_aug_dat['label'] = train_aug_dat['label'].replace('negative', 1)
            total_src_list['train'] = total_src_list['train'] + list(map(train_aug_dat['aug_text'].tolist().__getitem__, sampling_ix))
            total_trg_list['train'] = total_trg_list['train'] + list(map(train_aug_dat['label'].tolist().__getitem__, sampling_ix))

        # 2) Valid data load
        # total_src_list['valid'] = np.array(dataset['test']['text'])[origin_index]
        # total_trg_list['valid'] = np.array(dataset['test']['label'])[origin_index]
        total_src_list['valid'] = np.array(dataset['test']['text'])
        total_trg_list['valid'] = np.array(dataset['test']['label'])

        # 3) Test data load
        # total_src_list['test'] = np.array(dataset['test']['text'])[split_index]
        # total_trg_list['test'] = np.array(dataset['test']['label'])[split_index]
        total_src_list['test'] = np.array(dataset['test']['text'])
        total_trg_list['test'] = np.array(dataset['test']['label'])

    return total_src_list, total_trg_list
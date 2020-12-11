# refer to
# https://www.cs.toronto.edu/~kriz/cifar.html

try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import pickle
import numpy as np
import shutil
from numpy import array

url_base = 'http://www.cs.utoronto.ca/~kriz/'
key_file = {
    '1': 'data_batch_1',
    '2': 'data_batch_2',
    '3': 'data_batch_3',
    '4': 'data_batch_4',
    '5': 'data_batch_5',
    'test_data_set': 'test_batch',
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
train_data_file = dataset_dir + "/cifar-10-batches-py/"
test_data_file = dataset_dir + "/cifar-10-batches-py/test_batch"

cifar10 = 'cifar-10-python.tar.gz'
save_file = dataset_dir + "/" + cifar10


def _download(file_name):
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")


def convert_bytes_to_string(data):
    if isinstance(data, bytes):
        return data.decode('ascii')
    if isinstance(data, dict):
        return dict(map(convert_bytes_to_string, data.items()))
    if isinstance(data, tuple):
        return map(convert_bytes_to_string, data)

    return data


def _normalize(dataset):
    for key in dataset:
        if key == 'data':
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    return dataset


def _change_one_hot_label(x):
    t = np.zeros((len(x), 10))
    for idx, row in enumerate(t):
        row[x[idx]] = 1

    return t


def init_cifar10():
    _download(cifar10)

    if os.path.exists(save_file):
        shutil.unpack_archive(save_file)


def load_cifar10(normalize=True, flatten=True, one_hot_label=True, data_batch_number='1'):

    init_cifar10()

    with open(train_data_file + key_file[data_batch_number], 'rb') as fo:
        train_data_set = pickle.load(fo, encoding='bytes')
    train_data_set = convert_bytes_to_string(train_data_set)

    with open(test_data_file, 'rb') as fo:
        test_data_set = pickle.load(fo, encoding='bytes')
    test_data_set = convert_bytes_to_string(test_data_set)

    train_data_set['labels'] = array(train_data_set['labels'])
    test_data_set['labels'] = array(test_data_set['labels'])

    if normalize:
        train_data_set = _normalize(train_data_set)
        test_data_set = _normalize(test_data_set)

    if not flatten:
        train_data_set['data'] = train_data_set['data'].reshape(-1, 3, 32, 32)
        test_data_set['data'] = test_data_set['data'].reshape(-1, 3, 32, 32)

    if one_hot_label:
        train_data_set['labels'] = _change_one_hot_label(train_data_set['labels'])
        test_data_set['labels'] = _change_one_hot_label(test_data_set['labels'])

    return (train_data_set['data'], train_data_set['labels']), (test_data_set['data'], test_data_set['labels'])

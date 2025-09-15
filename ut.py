import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim

import random

from sklearn.metrics import accuracy_score
from tslearn.clustering import TimeSeriesKMeans
import pandas as pd

def plot(x, colors, domain):

    custom_colors = ['#E67E22', '#2980B9', '#7F00FD', '#D4AF37', '#C5767B', '#DC2626', '#C71585', '#8B4513', '#556B2F', '#64FBC4']

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'

    f = plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.subplot()

    sc = ax.scatter(
        x[:, 0],
        x[:, 1],
        lw=0.5,
        s=80,
        edgecolors='w',
        c=[custom_colors[i] for i in colors.astype(np.int8)]
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    plt.savefig(
        f'./{domain}.pdf',
        dpi=600,
        bbox_inches='tight',
        pad_inches=0.05,  # 减少周围空白
        metadata={'CreationDate': None}  # 移除时间戳保证可重复性
    )
    plt.savefig(
        f'./{domain}.png',
        dpi=600,
        bbox_inches='tight'
    )
    plt.close()
    return f, ax, []


def adjust_labels(y):
    min_val = np.min(y)
    if min_val != 0:
        return y - min_val
    return y

def filter_classes_by_count(dataset,X_train, y_train, keep_top_n=5, mid_count=30, percentage=0.5,filter=True):

    if filter:
        if dataset == 'CharacterTrajectories':
            keep_top_n = 5
            mid_count = 20
            percentage = 0.1
        elif dataset == 'PenDigits':
            keep_top_n = 3
            mid_count = 100
            percentage = 0.1
        elif dataset == 'SpokenArabicDigits':
            keep_top_n = 3
            mid_count = 100
            percentage = 0.02
        elif dataset == 'InsectWingbeat':
            keep_top_n = 3
            mid_count = 500
            percentage = 0.1
        elif dataset == 'Crop':
            keep_top_n = 7
            mid_count = 100
            percentage = 0.1
        elif dataset == 'TwoPatterns':
            keep_top_n = 1
            mid_count = 90
            percentage = 0.1
        elif dataset == 'MelbournePedestrian':
            keep_top_n = 3
            mid_count = 50
            percentage = 0.1
        else:
            filter = False
    else:
        filter = False

    min_label = np.min(y_train)

    if min_label > 0:
        y_train = y_train - 1
    if filter:

        class_counts = np.bincount(y_train)
        sorted_classes = np.argsort(class_counts)[::-1]

        top_classes = sorted_classes[:keep_top_n]
        other_classes = sorted_classes[keep_top_n:]


        max_top_count = np.max(class_counts[top_classes])


        last_limit = max(1, int(max_top_count * percentage))




        filtered_indices = []

        for c in top_classes:
            class_indices = np.where(y_train == c)[0]
            filtered_indices.extend(class_indices)

        for i, c in enumerate(other_classes):
            class_indices = np.where(y_train == c)[0]

            if i < len(other_classes) - 1:
                limit = mid_count

            else:
                limit = last_limit

            if len(class_indices) > limit:
                selected_indices = np.random.choice(class_indices, limit, replace=False)
                filtered_indices.extend(selected_indices)
            else:
                filtered_indices.extend(class_indices)

        X_filtered = X_train[filtered_indices]
        y_filtered = y_train[filtered_indices]
    else:
        X_filtered = X_train
        y_filtered = y_train

    return X_filtered, y_filtered


def modify_labels(y, modify_percentage, num_classes=None, random_state=None):


    if random_state is not None:
        np.random.seed(random_state)

    y = np.array(y)


    if num_classes is None:
        num_classes = np.max(y) + 1


    y_modified = y.copy()
    total_modified = 0

    for c in range(num_classes):

        class_indices = np.where(y == c)[0]
        class_size = len(class_indices)

        if class_size == 0 or class_size <10:
            continue


        num_modify = max(1, int(class_size * modify_percentage)) if modify_percentage > 0 else 0

        num_modify = min(num_modify, class_size)


        modify_indices = np.random.choice(class_indices, num_modify, replace=False)


        for idx in modify_indices:

            possible_labels = [label for label in range(num_classes) if label != c]
            new_label = np.random.choice(possible_labels)
            y_modified[idx] = new_label


        total_modified += num_modify



    total_samples = len(y)
    actual_modify_percentage = 100.0 * total_modified / total_samples



    return y_modified


def get_data_and_label_from_csv_file(file_path, label_path):
    y = None

    time_data = pd.read_csv(file_path, header=None)
    time_label = pd.read_csv(label_path, header=None)
    y = np.squeeze(time_label.values)
    X = np.expand_dims(time_data.values, axis=1)

    assert (y is not None)
    assert (y.shape[0] == X.shape[0])
    return X, y


def UCR_multivariate_data_loader(dataset_path, dataset_name):
    Train_dataset_path = dataset_path + '/' + dataset_name + '/' + 'TRAIN.csv'
    Test_dataset_path = dataset_path + '/' + dataset_name + '/' + 'TEST.csv'
    Train_label_path = dataset_path + '/' + dataset_name + '/' + 'TRAIN_label.csv'
    Test_label_path = dataset_path + '/' + dataset_name + '/' + 'TEST_label.csv'
    X_train, y_train = get_data_and_label_from_csv_file(Train_dataset_path, Train_label_path)
    X_test, y_test = get_data_and_label_from_csv_file(Test_dataset_path, Test_label_path)

    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test

def sample_ts_segments(X, shapelets_size, n_segments=10000):

    n_ts, n_channels, len_ts = X.shape
    samples_i = random.choices(range(n_ts), k=n_segments)
    segments = np.empty((n_segments, n_channels, shapelets_size))
    for i, k in enumerate(samples_i):
        s = random.randint(0, len_ts - shapelets_size)
        segments[i] = X[k, :, s:s+shapelets_size]
    return segments


def get_weights_via_kmeans(X, shapelets_size, num_shapelets, n_segments=10000):

    segments = sample_ts_segments(X, shapelets_size, n_segments).transpose(0, 2, 1)
    k_means = TimeSeriesKMeans(n_clusters=num_shapelets, metric="euclidean", max_iter=50).fit(segments)
    clusters = k_means.cluster_centers_.transpose(0, 2, 1)
    return clusters


class MaskBlock(nn.Module):
    def __init__(self, p=0.1):
        super(MaskBlock, self).__init__()
        
        self.net = nn.Dropout(p=p)
    def forward(self, X):
        return self.net(X)



class LinearBlock(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(LinearBlock, self).__init__()
        
        #self.linear = nn.Sequential(nn.Linear(in_channel, 256), nn.ReLU(), nn.Linear(256, n_classes))
        self.linear = nn.Linear(in_channel, n_classes)
    
    def forward(self, X):
        return self.linear(X)

class LinearClassifier():
    def __init__(self, in_channel, n_classes, batch_size=256, lr=1e-3, wd=1e-4, max_epoch=200):
        super(LinearClassifier, self).__init__()
        
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.lr = lr
        
        self.wd = wd
        self.max_epoch = max_epoch
        
        self.net = LinearBlock(in_channel, n_classes)
        
    
    def train(self, X, y):
        X = torch.from_numpy(X)
        X = X.float()
        
        y = torch.from_numpy(y)
        y = y.long()
        
        self.net.cuda()
        
        # loss, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=200, min_lr=0.0001)
        
        # build dataloader
        train_dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=max(int(min(X.shape[0], self.batch_size)), 4), shuffle=True)
        
        
        
        
        self.net.train()
        
        for epoch in range(self.max_epoch):
            losses = []
            for (x, y) in train_loader:
                x = x.cuda()
                y = y.cuda()
                logits = self.net(x)
                loss = criterion(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            scheduler.step(loss)
                
    
    def predict(self, X):
        X = torch.from_numpy(X)
        X = X.float()
        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(dataset, batch_size=max(int(min(X.shape[0], self.batch_size)), 4), shuffle=False)
        
        predict_list = np.array([])
        
        self.net.eval()
        
        for (x, ) in loader:
            x = x.cuda()
            y_predict = self.net(x)
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
        
        return predict_list



def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def z_normalize(a, eps=1e-7):
    return (a - np.mean(a, axis=-1, keepdims=True)) / (eps + np.std(a, axis=-1, keepdims=True))



def replace_nan_with_near_value(a):
    mask = np.isnan(a)
    idx = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    out = a[np.arange(idx.shape[0])[:,None], idx]
    return np.float32(out)

def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a


def fill_out_with_Nan(data,max_length):
    #via this it can works on more dimensional array
    pad_length = max_length-data.shape[-1]
    if pad_length == 0:
        return data
    else:
        pad_shape = list(data.shape[:-1])
        pad_shape.append(pad_length)
        Nan_pad = np.empty(pad_shape)*np.nan
        return np.concatenate((data, Nan_pad), axis=-1)
    

def get_label_dict(file_path):
    label_dict ={}
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            if '@classLabel' in line:
                label_list = line.replace('\n','').split(' ')[2:]
                for i in range(len(label_list)):
                    label_dict[label_list[i]] = i 
                
                break
    return label_dict


def get_data_and_label_from_ts_file(file_path,label_dict):
    with open(file_path) as file:
        lines = file.readlines()
        Start_reading_data = False
        Label_list = []
        Data_list = []
        max_length = 0
        for line in lines:
            if Start_reading_data == False:
                if '@data'in line:
                    Start_reading_data = True
            else:
                temp = line.split(':')
                Label_list.append(label_dict[temp[-1].replace('\n','')])
                data_tuple= [np.expand_dims(np.fromstring(channel, sep=','), axis=0) for channel in temp[:-1]]
                max_channel_length = 0
                for channel_data in data_tuple:
                    if channel_data.shape[-1]>max_channel_length:
                        max_channel_length = channel_data.shape[-1]
                data_tuple = [fill_out_with_Nan(data,max_channel_length) for data in data_tuple]
                data = np.expand_dims(np.concatenate(data_tuple, axis=0), axis=0)
                Data_list.append(data)
                if max_channel_length>max_length:
                    max_length = max_channel_length
        
        Data_list = [fill_out_with_Nan(data,max_length) for data in Data_list]
        X =  np.concatenate(Data_list, axis=0)
        Y =  np.asarray(Label_list)
        
        return np.float32(X), Y




def TSC_multivariate_data_loader(dataset_path, dataset_name):
    
    Train_dataset_path = dataset_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN.ts'
    Test_dataset_path = dataset_path + '/' + dataset_name + '/' + dataset_name + '_TEST.ts'
    label_dict = get_label_dict(Train_dataset_path)
    X_train, y_train = get_data_and_label_from_ts_file(Train_dataset_path,label_dict)
    X_test, y_test = get_data_and_label_from_ts_file(Test_dataset_path,label_dict)
    
    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test

def generate_binomial_mask(size, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=size)).cuda()


def eval_accuracy(model, X, Y, X_test, Y_test, normalize=False, lr=1e-3, wd=1e-4):
    transformation = model.transform(X, result_type='numpy', normalize=normalize)
    clf = LinearClassifier(transformation.shape[1], len(set(Y)), lr=lr, wd=wd)
    clf.train(transformation, Y)
    acc_train = accuracy_score(clf.predict(transformation), Y)
    acc_test = accuracy_score(clf.predict(model.transform(X_test, result_type='numpy', normalize=normalize)), Y_test)
    return acc_train, acc_test
import os
import sys

from matplotlib import pyplot as plt
from matplotlib.pyplot import get
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.optim as optim
from sklearn.mixture import GaussianMixture



import copy

import random
import numpy as np
import torch.nn.functional as F
from math import inf
import string
import random
from sklearn import metrics
import torch.nn as nn




def set_tensor(tensor_var, boolen, device):
    # print(tensor_var)
    tensor_var = tensor_var.to(device)
    # tensor_var = tensor_var.to(device,non_blocking=True)
    #return Variable(tensor_var, requires_grad=boolen)
    tensor_var.requires_grad = boolen
    return tensor_var


def get_knn_center(model,dataloader_,device):

    feats_all = torch.tensor([],device=device)
    with torch.no_grad():
        for idxs,(inputs, labels,_)  in enumerate(dataloader_):

            inputs = set_tensor(inputs,False,device).type(torch.float)
            labels = set_tensor(labels,False,device).type(torch.long)

            features,_ = model(inputs,False)

            feats_all = torch.cat([feats_all,features],dim=0)


    featmean = feats_all.mean(0)


    return {'mean': featmean}



def bi_dimensional_sample_selection_2(args, model, loader, epoch,y_ori, devices):
    model.eval()

    select_sample_index_list = []
    select_sample_prob_list = []

    warmup_trainloader = loader
    cfeat = get_knn_center(model, warmup_trainloader, devices)
    avg_pred_list = get_avg_pred_list_3(args,warmup_trainloader, model, devices)

    avg_pred_list_2 = get_avg_pred_list_4(args,warmup_trainloader, avg_pred_list, model, devices)

    mean_feat = cfeat['mean']

    centriod_list, sample_num_list = get_adaptive_centriod_2(args, warmup_trainloader, avg_pred_list_2, mean_feat,
                                                             model, devices)

    centriod_distance = torch.softmax(torch.einsum('ij,jk->ik', centriod_list, centriod_list.T), dim=1)

    wjsd_infos, index_infos, targets = get_wjsd_info_2(args,warmup_trainloader, avg_pred_list, model, devices)

    acd_infos = get_adaptive_centriod_distance_info_2(warmup_trainloader, centriod_list, model, mean_feat, devices)
    for class_num in range(args.num_classes):
        # bi_dimensional_sample_separation

        select_index = targets == class_num

        index_info = index_infos[select_index].cpu().numpy()
        wjsd_info = wjsd_infos[select_index].cpu().numpy()
        wjsd_info = (wjsd_info - wjsd_info.min()) / (wjsd_info.max() - wjsd_info.min())
        acd_info = acd_infos[select_index].cpu().numpy()

        # 提取非NaN的有效值
        valid_values = acd_info[~np.isnan(acd_info)]

        if len(valid_values) > 0:
            random_valid = np.random.choice(valid_values, size=acd_info.shape)
            acd_info = np.where(np.isnan(acd_info), random_valid, acd_info)
        else:
            acd_info = np.zeros_like(acd_info)

        acd_info = (acd_info - acd_info.min()) / (acd_info.max() - acd_info.min())

        combine_wjsd = wjsd_info.reshape(-1, 1)
        combine_acd = acd_info.reshape(-1, 1)

        prob_wjsd, gmm_wjsd = gmm_fit_func(combine_wjsd)

        try:
            prob_acd, gmm_acd = gmm_fit_func(combine_acd)
        except Exception as e:
            prob_acd = prob_wjsd.copy()
            gmm_acd = gmm_wjsd

        cluster_select_index_1_wjsd = (prob_wjsd[:, gmm_wjsd.means_.argmin()] > 0.5)
        cluster_select_index_2_wjsd = ~cluster_select_index_1_wjsd
        cluster_index_1_wjsd = index_info[cluster_select_index_1_wjsd]
        cluster_index_2_wjsd = index_info[cluster_select_index_2_wjsd]

        cluster_select_index_1_acd = (prob_acd[:, gmm_acd.means_.argmin()] > 0.5)
        cluster_select_index_2_acd = ~cluster_select_index_1_acd
        cluster_index_1_acd = index_info[cluster_select_index_1_acd]
        cluster_index_2_acd = index_info[cluster_select_index_2_acd]

        acd_wjsd_mean_1 = wjsd_info[cluster_select_index_1_acd].mean(0)
        acd_wjsd_std_1 = wjsd_info[cluster_select_index_1_acd].std(0)
        acd_wjsd_pred_1 = gmm_wjsd.predict(acd_wjsd_mean_1.reshape(1, -1))[0]

        acd_wjsd_mean_2 = wjsd_info[cluster_select_index_2_acd].mean(0)
        acd_wjsd_std_2 = wjsd_info[cluster_select_index_2_acd].std(0)
        acd_wjsd_pred_2 = gmm_wjsd.predict(acd_wjsd_mean_2.reshape(1, -1))[0]

        std_list = [1] * 2
        std_list[acd_wjsd_pred_1] = acd_wjsd_std_1
        std_list[acd_wjsd_pred_2] = acd_wjsd_std_2
        x = 1
        if x<0 and (acd_wjsd_pred_1 == acd_wjsd_pred_2 and acd_wjsd_pred_1 != gmm_wjsd.means_.argmin()) or (
                std_list[gmm_wjsd.means_.argmax()] / (std_list[gmm_wjsd.means_.argmin()]+ 1e-8) < 0.65):
            select_sample_index_list.extend(cluster_index_1_wjsd)
            select_sample_prob_list.extend(prob_wjsd[:, gmm_wjsd.means_.argmin()][cluster_select_index_1_wjsd])
        else:
            centriod_distance_copy = copy.deepcopy(centriod_distance[class_num])
            current_centriod_distance = copy.deepcopy(centriod_distance_copy[class_num])
            centriod_distance_copy[class_num] = 0
            max_centriod_distance, max_centriod_indice = centriod_distance_copy.topk(k=1, largest=True)
            if (abs((current_centriod_distance - max_centriod_distance[
                0]).item()) < 0.1 * current_centriod_distance.item() and (
                    sample_num_list[class_num] < sample_num_list[max_centriod_indice[0].item()])):
                select_sample_index_list.extend(cluster_index_1_acd)
                select_sample_prob_list.extend(prob_acd[:, gmm_acd.means_.argmin()][cluster_select_index_1_acd])
            else:

                select_sample_index_list.extend(cluster_index_2_acd)
                select_sample_prob_list.extend(prob_acd[:, gmm_acd.means_.argmax()][cluster_select_index_2_acd])

    return select_sample_index_list, select_sample_prob_list


def get_normalization_info(info_1, info_2):
    info = np.array(info_1 + info_2)
    normal_info = (info - info.min()) / (info.max() - info.min())
    normal_info_1 = normal_info[:len(info_1)].tolist()
    normal_info_2 = normal_info[len(info_1):].tolist()
    return normal_info_1, normal_info_2

class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)



def get_adaptive_centriod_distance_info(data_loader, centriod, model, meat_feat, devices):
    model.eval()
    dist_info = []
    index_info = []
    for i, (input, target, indexs) in enumerate(data_loader):
        input_var = set_tensor(input, False, devices)
        cur,_ = model(input_var,False)
        features = cur - meat_feat
        features = F.normalize(features, p=2, dim=1)

        dist = torch.einsum('ij,j->i', features, centriod.T)
        dist_info.extend(dist.tolist())
        index_info.extend(indexs.tolist())
    return dist_info, np.array(index_info)


def get_adaptive_centriod_distance_info_(data_loader, centriod, model, meat_feat, devices):
    model.eval()
    dist_info = []
    index_info = []
    feature_info = []
    for i, (input, target, indexs) in enumerate(data_loader):
        input_var = set_tensor(input, False, devices)
        cur,_ = model(input_var,False)
        features_ = cur - meat_feat
        features = F.normalize(features_, p=2, dim=1)

        dist = torch.einsum('ij,j->i', features, centriod.T)
        dist_info.extend(dist.tolist())
        index_info.extend(indexs.tolist())
        feature_info.extend(features_.tolist())
    return dist_info, np.array(index_info), np.array(feature_info)


def get_adaptive_centriod(args, global_dataloader, local_dataloader, avg_pred, class_num, feat_mean, model, devices):
    high_confidence_samples, sample_num = get_high_confidence_samples(local_dataloader, avg_pred, class_num, model,
                                                                      devices)

    adptive_feat_c = high_confidence_samples - feat_mean
    adptive_feat_cl2 = F.normalize(adptive_feat_c, p=2, dim=1)
    adptive_centriod = adptive_feat_cl2.mean(0)
    return adptive_centriod, sample_num


def get_high_confidence_samples(global_dataloader, avg_pred, class_num, model, devices):
    select_features_list = torch.tensor([]).to(devices)
    avg_pred = avg_pred[class_num]
    sample_num = 0
    for i, (input, target, indexs) in enumerate(global_dataloader):
        input_var = set_tensor(input, False, devices)

        features,y_f = model(input_var,False)
        preds = torch.softmax(y_f, dim=1)
        arg_idx = torch.argmax(preds, dim=1)
        select_ = torch.eq(arg_idx, torch.argmax(avg_pred))
        get_high_confidence_criterion = avg_pred[torch.argmax(avg_pred)]
        select_index = torch.gt(preds[:, torch.argmax(avg_pred)], get_high_confidence_criterion)

        select_features = features[select_index * select_]
        sample_num += (select_index * select_).sum().item()
        select_features_list = torch.cat([select_features_list, select_features], dim=0)

    if sample_num == 0:
        for i, (input, target, indexs) in enumerate(global_dataloader):
            input_var = set_tensor(input, False, devices)

            features,_ = model(input_var,False)

            select_features = features
            sample_num += len(target)
            select_features_list = torch.cat([select_features_list, select_features], dim=0)

    return select_features_list, sample_num


def get_avg_pred(data_loader, model, devices):
    model.eval()

    avg_pred = torch.tensor([]).to(devices)
    for i, (input, target, indexs) in enumerate(data_loader):
        input_var = set_tensor(input, False, devices)

        _,y_f = model(input_var,False)
        out = torch.softmax(y_f, dim=1).mean(0).unsqueeze(0)
        avg_pred = torch.cat([avg_pred, out], dim=0)

    return avg_pred.mean(0)


def get_avg_pred_2(data_loader, avg_pred_2, model, device):
    model.eval()
    avg_pred = torch.tensor([]).to(device)
    avg_argmax = torch.argmax(avg_pred_2, dim=0)
    for i, (input, target, indexs) in enumerate(data_loader):
        input_var = set_tensor(input, False, device)

        _,y_f = model(input_var,False)
        out = torch.softmax(y_f, dim=1)
        idx = [i for i in range(target.shape[0])]
        weight = torch.clamp(out[idx, avg_argmax] / avg_pred_2[avg_argmax], min=1)
        out[idx, avg_argmax] = weight * out[idx, avg_argmax]
        avg_pred = torch.cat([avg_pred, out.mean(0).unsqueeze(0)], dim=0)

    return avg_pred.mean(0)


def get_avg_pred_list(args, loader, model, devices):
    avg_pred_list = torch.tensor([]).to(devices)
    for class_num in range(args.num_classes):
        class_dataloader = loader.run(mode='single', class_num=class_num)
        avg_pred = get_avg_pred(class_dataloader, model, devices).unsqueeze(0)

        avg_pred_list = torch.cat([avg_pred_list, avg_pred], dim=0)
        del class_dataloader
    return avg_pred_list


def get_avg_pred_list_2(args, loader, avg_pred_list, model, devices):
    avg_pred_list_2 = torch.tensor([]).to(devices)
    for class_num in range(args.num_classes):
        class_dataloader = loader.run(mode='single', class_num=class_num)
        avg_pred = get_avg_pred_2(class_dataloader, avg_pred_list[class_num], model, devices).unsqueeze(0)

        avg_pred_list_2 = torch.cat([avg_pred_list_2, avg_pred], dim=0)
        del class_dataloader
    return avg_pred_list_2






def kl_divergence(p, q):
    return (p * ((p + 1e-10) / (q + 1e-10)).log()).sum(dim=1)


def gmm_fit_func(input_loss):
    input_loss = np.array(input_loss)
    has_nan = np.isnan(input_loss).any()
    has_inf = np.isinf(input_loss).any()
    max_float32 = np.finfo(np.float32).max  # float32的最大值
    has_large = (input_loss > max_float32).any()

    if has_nan or has_inf or has_large:

        valid_mask = ~(np.isnan(input_loss) | np.isinf(input_loss) | (input_loss > max_float32))
        input_loss_clean = input_loss[valid_mask]

        if len(input_loss_clean) == 0:
            raise ValueError("请检查input_loss的生成逻辑")
    else:
        input_loss_clean = input_loss  

    # 4. 拟合GMM
    gmm = GaussianMixture(n_components=2, max_iter=30, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss_clean.reshape(-1, 1))  # GMM要求输入为二维数组(n_samples, n_features)
    prob = gmm.predict_proba(input_loss_clean.reshape(-1, 1))

    gmm = GaussianMixture(n_components=2, max_iter=30, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)

    return prob, gmm

def get_avg_pred_list_3(args,data_loader,model,devices):
    model.eval()

    preds = torch.tensor([],device=devices)
    targets = torch.tensor([],device=devices)
    avg_pred = torch.tensor([],device=devices)
    for i, (input, target,indexs) in enumerate(data_loader):

        input_var = set_tensor(input, False, devices)
        target_var = set_tensor(target, False, devices)
        _,y_f = model(input_var,False)
        out = torch.softmax(y_f,dim=1)
        preds = torch.cat([preds,out],dim=0)
        targets = torch.cat([targets,target_var],dim=0)

    for i in range(args.num_classes):
        avg_pred = torch.cat([avg_pred,preds[targets == i].mean(0).unsqueeze(0)],dim=0)
    return avg_pred

def get_avg_pred_list_4(args,data_loader,avg_pred_2,model,devices):
    model.eval()
    avg_pred = torch.tensor([],device=devices)# .to(device)
    avg_argmax = torch.argmax(avg_pred_2,dim=1)
    preds = torch.tensor([],device=devices)
    targets = torch.tensor([],device=devices)
    for i, (input, target,indexs) in enumerate(data_loader):

        input_var = set_tensor(input, False, devices)
        target_var = set_tensor(target, False, devices)

        _,y_f = model(input_var,False)
        out = torch.softmax(y_f,dim=1)
        idx = [i for i in range(target.shape[0])]
        weight = torch.clamp(out[idx,avg_argmax[target_var]] / avg_pred_2[target_var,avg_argmax[target_var]],min=1)
        out[idx,avg_argmax[target_var]] = weight * out[idx,avg_argmax[target_var]]
        preds = torch.cat([preds,out],dim=0)
        targets = torch.cat([targets,target_var],dim=0)
    for i in range(args.num_classes):
        avg_pred = torch.cat([avg_pred,preds[targets == i].mean(0).unsqueeze(0)],dim=0)
    return avg_pred



def get_wjsd_info_2(args,data_loader,avg_pred,model,device):
    model.eval()

    JS_dist = Jensen_Shannon()
    targets = torch.tensor([],device=device)
    jsd_info = torch.tensor([],device=device)
    index_info = torch.tensor([],dtype=torch.long)
    for i, (input, target,indexs) in enumerate(data_loader):

        input_var = set_tensor(input, False, device)
        target_var = set_tensor(target, False, device)
        targets = torch.cat([targets,target_var],dim=0)
        _,y_f = model(input_var,False)
        out = torch.softmax(y_f,dim=1)

        idx = torch.tensor([x for x in range(len(out))])
        weight = out[idx,torch.argmax(out,dim=1)] / out[idx,target_var]

        weight_max = (avg_pred[target_var,torch.argmax(avg_pred[target_var],dim=1)] / avg_pred[target_var,target_var]).detach()
        weight_index  = weight > weight_max
        weight[weight_index] = weight_max[weight_index]

        jsd =  weight * JS_dist(out,  F.one_hot(target_var, num_classes = args.num_classes))

        jsd_info = torch.cat([jsd_info,jsd],dim=0)

        index_info = torch.cat([index_info,indexs],dim=0)
    return jsd_info,index_info,targets


def get_adaptive_centriod_2(args,local_dataloader,avg_pred,feat_mean,model,devices):

    adptive_centriod_list = torch.tensor([],device=devices)
    sample_num_list = []
    high_confidence_samples,targets = get_high_confidence_samples_2(local_dataloader,avg_pred,model,devices)

    adptive_feat_c = high_confidence_samples - feat_mean
    adptive_feat_cl2 = F.normalize(adptive_feat_c,p=2,dim=1)
    for i in range(args.num_classes):
        adptive_centriod_list = torch.cat([adptive_centriod_list,adptive_feat_cl2[targets == i].mean(0).unsqueeze(0)],dim=0)
        sample_num_list.append((targets == i).sum(0).item())

    return adptive_centriod_list,sample_num_list


def get_high_confidence_samples_2(global_dataloader, avg_pred, model, devices):
    select_features_list = torch.tensor([], device=devices)
    sample_num = 0
    targets = torch.tensor([], device=devices)
    for i, (input, target, indexs) in enumerate(global_dataloader):
        input_var = set_tensor(input, False, devices)
        target_var = set_tensor(target, False, devices)
        features,y_f = model(input_var,False)
        preds = torch.softmax(y_f, dim=1)
        arg_idx = torch.argmax(preds, dim=1)
        select_ = torch.eq(arg_idx, torch.argmax(avg_pred[target_var], dim=1))

        idx = [i for i in range(target_var.shape[0])]
        get_high_confidence_criterion = avg_pred[target_var, torch.argmax(avg_pred[target_var], dim=1)]
        select_index = torch.gt(preds[idx, torch.argmax(avg_pred[target_var], dim=1)], get_high_confidence_criterion)

        select_features = features[select_index * select_]

        select_features_list = torch.cat([select_features_list, select_features], dim=0)
        targets = torch.cat([targets, target_var[select_index * select_]], dim=0)
    return select_features_list, targets


def get_adaptive_centriod_distance_info_2(data_loader, centriod, model, meat_feat, devices):
    model.eval()
    dist_info = torch.tensor([], device=devices)

    for i, (input, target, indexs) in enumerate(data_loader):
        input_var = set_tensor(input, False, devices)
        target_var = set_tensor(target, False, devices)
        cur,_ = model(input_var,False)
        features = cur - meat_feat
        features = F.normalize(features, p=2, dim=1)

        dist = torch.einsum('ij,ji->i', features, centriod[target_var].T)

        dist_info = torch.cat([dist_info, dist], dim=0)

    return dist_info

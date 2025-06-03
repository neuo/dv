# -*- coding: utf-8 -*-

import os
import pickle

import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from preprocess import process_data_group

class DataGroup:
    def __init__(self, dir_name, diff22, diff32, diff23, diff1515, diff11):
        self.dir = dir_name  # 数据组的目录名
        self.diff22 = diff22  # 2mm 2% 误差
        self.diff32 = diff32  # 2mm 3% 误差
        self.diff23 = diff23  # 3mm 2% 误差
        self.diff1515 =  diff1515 # 1.5mm 1.5% 误差
        self.diff11 =  diff11 # 1mm 1% 误差
        self.adt_plan = None
        self.ref_plan = None
        self.ref_measure = None

    def __repr__(self):
        return f"Dir: {self.dir}, 2mm 2%: {self.diff22}, 2mm 3%: {self.diff32}, 3mm 2%: {self.diff23}"


def load_detail(dir, groups):
    # 假设 process_data_group 函数已经被导入并在外部定义
    for group in groups:  
        path = dir + group.dir
        ref_plan, ref_measure, adt_plan = process_data_group(path)
        group.adt_plan = adt_plan
        group.ref_plan = ref_plan
        group.ref_measure = ref_measure

    return groups
    

def init_data_structure(dir):
  # 读取 Excel 文件
  df = pd.read_excel(dir + "number.xlsx", sheet_name='Sheet1')

  # 读取第 2 列到第 5 列的数据
  data_groups = []
  for index, row in df.iterrows():
      if pd.notna(row.iloc[1]):  # 假设第 1 列是目录名，排除空值
          dir_name = str(row.iloc[1])
          diff22 = row.iloc[2]
          diff32 = row.iloc[3]
          diff23 = row.iloc[4]
          diff1515 = row.iloc[5]
          diff11 = row.iloc[6]
          data_groups.append(DataGroup(dir_name, diff22, diff32, diff23, diff1515, diff11))
  return data_groups

  from sklearn.model_selection import train_test_split

# 自定义数据集
class PredictDiffDataset(Dataset):
    def __init__(self, ref_plan_list, ref_measure_list, adt_plan_list, target_list):
        self.ref_plan_list = ref_plan_list
        self.ref_measure_list = ref_measure_list
        self.adt_plan_list = adt_plan_list
        self.target_list = target_list

    def __len__(self):
        return len(self.ref_plan_list)

    def __getitem__(self, idx):
        ref_plan = torch.tensor(self.ref_plan_list[idx], dtype=torch.float32)
        ref_measure = torch.tensor(self.ref_measure_list[idx], dtype=torch.float32)
        adt_plan = torch.tensor(self.adt_plan_list[idx], dtype=torch.float32)
        target = torch.tensor(self.target_list[idx], dtype=torch.float32)
        return ref_plan, ref_measure, adt_plan, target

def to_dataset(data_groups):
    # 假设 process_data_group 函数已经被导入并在外部定义
    ref_plan_list, ref_measure_list, adt_plan_list, target_list = [], [], [], [[], [], [], [], []]
    for group in data_groups:
        ref_plan_list.append(group.ref_plan)
        ref_measure_list.append(group.ref_measure)
        adt_plan_list.append(group.adt_plan)
        target_list[0].append(group.diff22) 
        target_list[1].append(group.diff32) 
        target_list[2].append(group.diff23) 
        target_list[3].append(group.diff1515) 
        target_list[4].append(group.diff11) 
    
    # 创建数据集
    return [
      PredictDiffDataset(ref_plan_list, ref_measure_list, adt_plan_list, target_list[0]),
      PredictDiffDataset(ref_plan_list, ref_measure_list, adt_plan_list, target_list[1]),
      PredictDiffDataset(ref_plan_list, ref_measure_list, adt_plan_list, target_list[2]),
      PredictDiffDataset(ref_plan_list, ref_measure_list, adt_plan_list, target_list[3]),
      PredictDiffDataset(ref_plan_list, ref_measure_list, adt_plan_list, target_list[4])
    ]

def split_data(dataset, train_size=0.7, val_size=0.15, test_size=0.15, random_state=142):
    """
    将数据集划分为训练集、验证集和测试集。
    
    参数:
    - dataset: 输入的数据集，通常是一个列表或 NumPy 数组。
    - train_size: 训练集的比例，默认 0.7。
    - val_size: 验证集的比例，默认 0.15。
    - test_size: 测试集的比例，默认 0.15。
    - random_state: 随机种子，保证结果可复现。
    
    返回:
    - train_data: 训练集。
    - val_data: 验证集。
    - test_data: 测试集。
    """
    
    # 确保训练集、验证集、测试集的比例之和为 1
    assert train_size + val_size + test_size == 1, "train_size + val_size + test_size 必须等于 1"

    # 首先划分出测试集
    train_data, test_data = train_test_split(dataset, test_size=test_size, random_state=random_state)

    # 接着从剩下的训练数据中划分出验证集
    val_ratio = val_size / (train_size + val_size)  # 在训练集+验证集的基础上划分验证集
    train_data, val_data = train_test_split(train_data, test_size=val_ratio, random_state=random_state)

    return train_data, val_data, test_data
    

def persist_cache(dir, obj):
    with open(dir + '/cache.pkl', 'wb') as f:
      pickle.dump(obj, f)

def restore_cache(dir):
    path = dir + '/cache.pkl'
    if not os.path.exists(path):
      return None
    with open(path, 'rb') as f:
      return pickle.load(f)

def init_data(dir):
  groups = restore_cache(dir)
  if not groups:
    groups = init_data_structure(dir)
    groups = load_detail(dir, groups)
    persist_cache(dir, groups)

  return groups

def init_data_for_train(dir, train_size=0.7, val_size=0.15, test_size=0.2):
  groups = init_data(dir)

  train_data, val_data, test_data = split_data(groups, train_size, val_size, test_size)

  for d in test_data:
    print(d.dir)

  return to_dataset(train_data), to_dataset(val_data), to_dataset(test_data)


def init_data_for_prediect(dir, remaining = []):
  groups = init_data(dir)

  subs = []
  for group in groups:
    if remaining and group.dir not in remaining:
      continue
    subs.append(group)

  return to_dataset(subs)















# vim: set ts=4 sw=4 sts=4 tw=100 et:

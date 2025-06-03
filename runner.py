# -*- coding: utf-8 -*-

# 安装依赖
# pip install pandas numpy scipy openpyxl scikit-learn

import torch.nn as nn

from loader import init_data_for_train
from check import calculate_errors_numpy, output_result
from process import train, predict
from model.cnn_transfrom import CNNTransformerModel
from model.cnn import CNNModel
from model.transformer import TransformerModel

from loss import WeightedMAELoss, MonotonicLoss, CombinedPearsonMAELoss

dir = './data/'

model = CNNTransformerModel()
# model = CNNModel()
# model = TransformerModel()

train_datasets, val_datasets, test_datesets = init_data_for_train(dir, 0.8, 0.1, 0.1)
descs = ["2mm 2%", "2mm 3%", "3mm 2%", "1.5mm 1.5%", "1mm 1%"]

for index in range(4, -1, -1):
  train_dataset = train_datasets[index]
  val_dataset = val_datasets[index]
  test_dateset = test_datesets[index]


  model_path = model.name + f'_{index}.pth'

  # criterion = CombinedPearsonMAELoss()
  criterion = nn.MSELoss()
  train(model, criterion, model_path, train_dataset, val_dataset)

  outputs, targets = predict(model, model_path, test_dateset)

  output_result(descs[index], outputs[0], targets[0])


















# vim: set ts=4 sw=4 sts=4 tw=100 et:

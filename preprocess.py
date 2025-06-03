# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator

def process_data_group(path):
  # 拼接出每个文件的路径
  adt_plan_file = path + '/ADT_plan.xlsx'
  ref_measure_file = path + '/Ref_measure.xlsx'
  ref_plan_file = path + '/Ref_plan.xlsx'

  # 第 1 步：定义用于加载和处理数据集的类
  class DatasetWithMaxValue:
      def __init__(self, file_path, precision):
          # 读取 Excel 文件并忽略空值（NaN）
          print(file_path)
          self.dataframe = pd.read_excel(file_path, sheet_name=0, index_col=0)
          self.dataframe.dropna(how='all', inplace=True)  # 删除整行为空的情况
          self.dataframe.dropna(axis=1, how='all', inplace=True)  # 删除整列为空的情况

          self.matrix = self.dataframe.values
          self.X_min = self.dataframe.columns.min()
          self.X_max = self.dataframe.columns.max()
          self.Y_min = self.dataframe.index.min()
          self.Y_max = self.dataframe.index.max()
          self.precision = precision
          self.maxValue = np.max(self.matrix)

      def normalize_by_max(self, reference_max_value):
          """通过给定的最大值对矩阵进行归一化"""
          self.matrix = self.matrix / reference_max_value if reference_max_value > 0 else self.matrix

      def adjust_to_ref_measure(self, ref_measure):
          """根据 Ref_measure 的范围调整矩阵"""
          X_min_ref = ref_measure.X_min
          X_max_ref = ref_measure.X_max
          Y_min_ref = ref_measure.Y_min
          Y_max_ref = ref_measure.Y_max

          # 根据 Ref_measure 的范围调整矩阵
          self.dataframe = self.dataframe.loc[
              (self.dataframe.index >= Y_min_ref) & (self.dataframe.index <= Y_max_ref),
              (self.dataframe.columns >= X_min_ref) & (self.dataframe.columns <= X_max_ref)
          ]

          self.matrix = self.dataframe.values
          self.X_min = self.dataframe.columns.min()
          self.X_max = self.dataframe.columns.max()
          self.Y_min = self.dataframe.index.min()
          self.Y_max = self.dataframe.index.max()

  # 第 2 步：加载并调整数据集
  ref_measure_with_max = DatasetWithMaxValue(ref_measure_file, precision=10)
  adt_plan_with_max = DatasetWithMaxValue(adt_plan_file, precision=1)
  ref_plan_with_max = DatasetWithMaxValue(ref_plan_file, precision=1)

  # 根据 Ref_measure 的范围调整 ADT_plan 和 Ref_plan
  adt_plan_with_max.adjust_to_ref_measure(ref_measure_with_max)
  ref_plan_with_max.adjust_to_ref_measure(ref_measure_with_max)

  # 第 3 步：对数据集进行归一化
  ref_measure_with_max.normalize_by_max(ref_measure_with_max.maxValue)
  ref_plan_with_max.normalize_by_max(ref_measure_with_max.maxValue)
  adt_plan_with_max.normalize_by_max(adt_plan_with_max.maxValue)

  # 第 4 步：对 Ref_measure 进行插值，将精度从 10mm 提升到 1mm
  def interpolate_to_higher_resolution(matrix, current_precision=10, target_precision=1):
      """
      对矩阵进行插值，将精度提高
      :param matrix: 输入的矩阵 (Ref_measure).
      :param current_precision: 矩阵的原始精度 (默认 10mm).
      :param target_precision: 插值后的目标精度 (默认 1mm).
      :return: 插值后的高分辨率矩阵.
      """
      rows, cols = matrix.shape
      x = np.linspace(0, (cols - 1) * current_precision, cols)
      y = np.linspace(0, (rows - 1) * current_precision, rows)

      interp_func = RegularGridInterpolator((y, x), matrix)

      x_new = np.linspace(0, (cols - 1) * current_precision, (cols - 1) * current_precision // target_precision + 1)
      y_new = np.linspace(0, (rows - 1) * current_precision, (rows - 1) * current_precision // target_precision + 1)

      x_new_grid, y_new_grid = np.meshgrid(x_new, y_new)

      interpolated_matrix = interp_func((y_new_grid, x_new_grid))

      return interpolated_matrix

  # 对 Ref_measure 进行插值
  ref_measure_interpolated_matrix_step4 = interpolate_to_higher_resolution(ref_measure_with_max.matrix)

  # 第 5 步：对 Ref_measure 和 Ref_plan 应用阈值
  threshold_ref_measure_step5 = 0.1
  ref_measure_interpolated_matrix_step5 = ref_measure_interpolated_matrix_step4.copy()

  # 对 Ref_measure 应用阈值
  ref_measure_interpolated_matrix_step5[ref_measure_interpolated_matrix_step5 < threshold_ref_measure_step5] = 0

  # 对 Ref_plan 进行相同的处理
  ref_plan_with_max.matrix[ref_measure_interpolated_matrix_step5 == 0] = 0

  # 对 ADT_plan 应用阈值
  threshold_adt_plan_step5 = 0.1
  adt_plan_with_max.matrix[adt_plan_with_max.matrix < threshold_adt_plan_step5] = 0

  # 保存最终的 3 个矩阵
  adt_plan = adt_plan_with_max.matrix
  ref_measure = ref_measure_interpolated_matrix_step5
  ref_plan = ref_plan_with_max.matrix

  return adt_plan, ref_measure, ref_plan


# vim: set ts=4 sw=4 sts=4 tw=100 et:

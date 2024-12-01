import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.preprocessing import MinMaxScaler
import json
from tqdm import tqdm
import warnings

from configx.configx import ConfigX

warnings.filterwarnings("ignore")


def merge_and_remove_duplicates(df_list, index_cols=['index', 'bugId']):
    for df in df_list:
        df.set_index(index_cols, inplace=True)
    df_combined = pd.concat(df_list, axis=1)
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    df_combined.reset_index(inplace=True)
    return df_combined

def check_columns_num(br,cf,relat):
    if br.shape[1] != 28: print("Attention: bugReportFeatures num!")
    if cf.shape[1] != 77: print("Attention: buggyFileFeatures num!")
    if relat.shape[1] != 18: print("Attention: RelationshhipFeatures num!")

def readSplitedData():
    file_fold = f"../data/splited_and_boosted_data/{dataset}"
    br_path = f"{file_fold}/bugReportsFeatures/"
    cf_path = f"{file_fold}/buggyFileFeatures/"
    relat_path = f"{file_fold}/relationFeatures/"

    fold_data = []
    fold_indices = []
    for i in range(5):
        br = pd.read_csv(f"{br_path}{i}.csv")
        cf = pd.read_csv(f"{cf_path}{i}.csv")
        relat =pd.read_csv(f"{relat_path}{i}.csv")
        check_columns_num(br,cf,relat)

        data = merge_and_remove_duplicates([br, cf, relat])

        # dropNa
        data = data.dropna()
        fold_data.append(data)
        fold_indices.extend([i] * len(data))
    df_combined = pd.concat(fold_data)
    return df_combined,fold_indices


def load_fold_data():
    df_combined,fold_indices = readSplitedData()
    drop_columns = ['index', 'bugId', 'rank_0', 'score_0', 'path_0', 'rank_1', 'path_1', 'score_1', 'rank_2', 'path_2', 'label']
    x_data = df_combined.drop(columns=drop_columns,axis=1)
    y_data = df_combined['label']

    print(f"当前训练数据量：{x_data.shape}")
    print(x_data.columns)


    # preprocess
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_data_normalized = scaler.fit_transform(x_data)
    #保留df格式
    x_data = pd.DataFrame(x_data_normalized, columns=x_data.columns, index=x_data.index)

    # Convert x_data and y_data to cupy arrays
    # x_data = cupy.asarray(x_data)  # Convert numpy array to cupy array
    # y_data = cupy.asarray(y_data)  # Convert numpy array to cupy array
    # x_data = cupy.asarray(x_data,dtype=cupy.float32)  # Convert numpy array to cupy array
    # y_data = cupy.asarray(y_data,dtype=cupy.int32)  # Convert numpy array to cupy array

    # 检查是否有非数值型数据
    print(f"Unique values in y_data: {np.unique(y_data)}")
    # 检查数据是否包含NaN
    print(f"NaN in x_data: {np.isnan(x_data).any()}")
    print(f"NaN in y_data: {np.isnan(y_data).any()}")
    print(x_data.head())
    print(x_data.shape)
    # x_data.to_csv(f"D:\\HitMore\\R_importance\\hitmore_all_features\\{dataset}_features.csv",index=False)
    merged_df = pd.concat([x_data, y_data], axis=1)
    print(merged_df.columns)

    merged_df.to_csv(f"D:\\HitMore\\R_importance\\hitmore_all_features\\{dataset}_features_withLabel.csv",index=False)

    return fold_indices, x_data , y_data

def calculateAvgMetrics(metricList):
    return sum(metricList)/len(metricList)

def main():
    # model_name_list = ['Random Forest','SVM','Logistic Regression']
    # for model_name in model_name_list:
    #     print("Current model： ", model_name)
    #     train_model(model_name)

    model_name = 'Random Forest'
    print("Current model： ", model_name)
    # train_model(model_name)


if __name__ == "__main__":
    configx = ConfigX()
    for dataset, file in configx.filepath_dict.items():
        if dataset == 'zookeeper':
            continue
        print(f"===dataset:{dataset}===")
        np.set_printoptions(threshold=np.inf)
        fold_indices, x_data , y_data = load_fold_data()
        # main()

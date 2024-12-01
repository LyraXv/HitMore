import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import json
from tqdm import tqdm
import warnings

from configx.configx import ConfigX

warnings.filterwarnings("ignore")


def merge_and_remove_duplicates(df_list, index_cols=['index', 'bugId']):
    # 设置指定的列为索引
    for df in df_list:
        df.set_index(index_cols, inplace=True)

    # 按索引合并多个 DataFrame
    df_combined = pd.concat(df_list, axis=1)

    # 移除重复的列
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]

    # 重置索引
    df_combined.reset_index(inplace=True)

    return df_combined

def check_columns_num(br,cf,relat):
    if br.shape[1] != 28: print("Attention: bugReportFeatures num!")
    if cf.shape[1] != 77: print("Attention: buggyFileFeatures num!")
    if relat.shape[1] != 18: print("Attention: RelationshhipFeatures num!")

def readSplitedData(): # onlyforgridCV
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

    # preprocess
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_data = scaler.fit_transform(x_data)

    return fold_indices, x_data , y_data


def train_model(model_name):
    if model_name not in models:
        print(f"模型 '{model_name}' 不存在，请选择以下模型之一: {list(models.keys())}")
        return

    model, params = models[model_name]
    # 使用 PredefinedSplit 固定五折划分
    ps = PredefinedSplit(test_fold=fold_indices)

    # 使用 Pipeline 将 StandardScaler、SMOTE 与 SVM 结合
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=0)),  # 过采样步骤
        # ('randomOverSampler', RandomOverSampler(random_state=42)),
        ('model', model)  # 模型步骤
    ])

    # 使用 GridSearchCV 进行网格搜索
    # scoring = ['accuracy', 'precision', 'recall', 'f1']
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted'
    }
    grid_search = GridSearchCV(pipeline, params, cv=ps, scoring=scoring, refit='f1', return_train_score=True, verbose=10,
                               n_jobs=-1)
    for _ in tqdm(range(1), desc=f'ML model {model_name} is training...'):
        grid_search.fit(x_data, y_data)

    all_results = []

    cv_results = grid_search.cv_results_

    for i in range(len(cv_results['params'])):
        res = [cv_results['params'][i], cv_results['mean_test_accuracy'][i], cv_results['mean_test_precision'][i],
               cv_results['mean_test_recall'][i], cv_results['mean_test_f1'][i]]
        all_results.append(res)

    print(all_results)

    # df_results = pd.DataFrame(all_results,
    #                           columns=['params', 'mean_test_accuracy', 'mean_test_precision', 'mean_test_recall',
    #                                    'mean_test_f1'])
    # df_results.to_csv(f"../data/splited_and_boosted_data/{dataset}/{dataset}_{model_name}_grid_search_results.csv")

    # 网格搜索训练后的副产品(以F1-score为标准)
    print("模型的最优参数：", grid_search.best_params_)
    print("最优模型分数：", grid_search.best_score_)
    print("最优模型对象：", grid_search.best_estimator_)


def main():
    # print(f"可选模型: {list(models.keys())}")
    # model_name = input("请输入要训练的模型名称: ")
    model_name = 'test'
    train_model(model_name)


if __name__ == "__main__":
    # Models and parameters
    models = {
        'SVM': (SVC(), [{'model__kernel': ['linear'], 'model__C': [2 ** x for x in range(-10, 11)]},
                        {'model__kernel': ['rbf', 'sigmoid'], 'model__C': [2 ** x for x in range(-10, 11)],
                         'model__gamma': [2 ** x for x in range(-10, 11)] + ['scale']},
                        {'model__kernel': ['poly'], 'model__C': [2 ** x for x in range(-10, 11)],
                         'model__gamma': [2 ** x for x in range(-10, 11)] + ['scale'], 'model__degree': [2, 3, 4, 5]}]),
        'Random Forest': (RandomForestClassifier(), {
            'model__max_features': [x / 10 for x in range(1, 11)] + ['sqrt', 'log2'],
            'model__max_samples': [x / 10 for x in range(1, 11)], 'model__n_jobs': [20]}),
        'Naive Bayes': (GaussianNB(), {'model__var_smoothing': [x / 10 for x in range(1, 11)] + [1e-09]}),
        'Logistic Regression': (LogisticRegression(), {'model__C': [x for x in range(1, 11)]}),
        'Decision Tree': (DecisionTreeClassifier(), {'model__min_samples_split': [x for x in range(1, 61)],
                                                     'model__min_samples_leaf': [x for x in range(1, 61)]}),
        'test': (DecisionTreeClassifier(), {'model__min_samples_split': [60],
                                                     'model__min_samples_leaf': [41]})
    }

    configx = ConfigX()
    for dataset, file in configx.filepath_dict.items():
        # dataset = 'lucene'
        if dataset != 'lucene':
            continue
        print(f"===dataset:{dataset}===")
        np.set_printoptions(threshold=np.inf)
        fold_indices, x_data , y_data = load_fold_data()
        main()

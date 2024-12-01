'''
读数据（按类型读）
调参(对所有数据集进行联合超参数调优？）
训练
'''
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from configx.configx import ConfigX


def preprocessData():
    return None


def merge_and_remove_duplicates(df_list, index_cols =['index','bugId']):
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

def readSplitedData(path,fold_idx):
    train_files = [f"{path}{i}.csv" for i in range(5) if i != fold_idx]
    test_file = f"{path}{fold_idx}.csv"
    train_data = pd.concat([pd.read_csv(file) for file in train_files])
    test_data = pd.read_csv(test_file)
    return train_data,test_data


def load_fold_data(fold_idx,dim = 'all'): # fold_num as test
    file_fold = f"../data/splited_and_boosted_data/{dataset}"
    br_path = f"{file_fold}/bugReportsFeatures/"
    cf_path = f"{file_fold}/buggyFileFeatures/"
    relat_path = f"{file_fold}/relationFeatures/"
    if dim == 'br':
        train_data, test_data = readSplitedData(br_path,fold_idx)
    elif dim =='cf':
        train_data, test_data = readSplitedData(cf_path,fold_idx)
    elif dim == 'relat':
        train_data, test_data = readSplitedData(relat_path,fold_idx)
    else:
        # train_br, test_br = readSplitedData(br_path,fold_idx)
        train_cf, test_cf = readSplitedData(cf_path,fold_idx)
        train_relat, test_relat = readSplitedData(relat_path,fold_idx)

        train_data = merge_and_remove_duplicates([train_cf,train_relat])
        test_data = merge_and_remove_duplicates([test_cf,test_relat])
        # train_data = merge_and_remove_duplicates([train_br,train_cf,train_relat])
        # test_data = merge_and_remove_duplicates([test_br,test_cf,test_relat])

    x_train = train_data.drop('label', axis=1) # wait to adjust
    y_train = train_data['label']
    x_test = test_data.drop('label', axis=1)
    y_test = test_data['label']

    preprocessData(x_train,y_train)


    return x_train, y_train, x_test, y_test



def train_and_evaluate(data_dim):
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'SVM': SVC(kernel='rbf', C=1, probability=True, random_state=42),
        'NaiveBayes': GaussianNB()
    }

    results = {name: {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []} for name in models}

    # read data by dimension type(注意数据是五折划分的！）
    for i in range(5):
        train_x, train_y, test_x, test_y = load_fold_data(i,data_dim)
        for name, model in models.items():
            model.fit(train_x, train_y)
            y_pred = model.predict(test_x)
            y_proba = model.predict_proba(test_y)[:, 1] if hasattr(model, "predict_proba") else None

            results[name]['accuracy'].append(accuracy_score(test_y, y_pred))
            results[name]['precision'].append(precision_score(test_y, y_pred, average='macro'))
            results[name]['recall'].append(recall_score(test_y, y_pred, average='macro'))
            results[name]['f1'].append(f1_score(test_y, y_pred, average='macro'))
            if y_proba is not None:
                results[name]['auc'].append(roc_auc_score(test_y, y_proba))

            # 保存模型
            joblib.dump(model, f'{name}_fold_{i}.pkl')
    average_results = {name: {metric: np.mean(scores) for metric, scores in metrics.items()} for name, metrics in
                       results.items()}
    # 打印结果
    for name, metrics in average_results.items():
        print(f"{name} Results:")
        for metric, score in metrics.items():
            print(f"{metric}: {score:.4f}")
        print("\n")


if __name__ == '__main__':
    configx = ConfigX()
    # pd.set_option('display.max_columns', None)
    for dataset, file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']:
            continue
        load_fold_data(0)
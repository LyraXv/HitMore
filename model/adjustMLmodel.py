
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


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
        if (train_data.shape[1] != 28): print("Attention: bugReportFeatures num!")
    elif dim =='cf':
        train_data, test_data = readSplitedData(cf_path,fold_idx)
        if (train_data.shape[1] != 77): print("Attention: buggyFileFeatures num!")
    elif dim == 'relat':
        train_data, test_data = readSplitedData(relat_path,fold_idx)
        if (train_data.shape[1] != 18): print("Attention: RelationshhipFeatures num!")
    else:
        train_br, test_br = readSplitedData(br_path,fold_idx)
        train_cf, test_cf = readSplitedData(cf_path,fold_idx)
        train_relat, test_relat = readSplitedData(relat_path,fold_idx)

        if (train_br.shape[1] != 28): print("Attention: bugReportFeatures num!")
        if (train_cf.shape[1] != 77): print("Attention: buggyFileFeatures num!")
        if (train_relat.shape[1] != 18): print("Attention: RelationshhipFeatures num!")

        train_data = merge_and_remove_duplicates([train_br,train_cf,train_relat])
        test_data = merge_and_remove_duplicates([test_br,test_cf,test_relat])

    x_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    x_test = test_data.drop('label', axis=1)
    y_test = test_data['label']

    x_data = pd.concat([x_train,x_test])
    y_data = pd.concat([y_train,y_test])

    drop_columns = ['index','bugId','rank_0','score_0','path_0','rank_1','path_1','score_1','rank_2','path_2']
    x_data = x_data.drop(columns=drop_columns,axis=1)

    # dropNa
    mask = x_data.notnull().all(axis=1)
    x_data= x_data[mask]
    y_data= y_data[mask]

    # preprocess
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)

    oversampler = SMOTE(random_state=0)
    os_features, os_labels = oversampler.fit_resample(x_data, y_data)

    return os_features,os_labels

def train_model(model_name):
    if model_name not in models:
        print(f"模型 '{model_name}' 不存在，请选择以下模型之一: {list(models.keys())}")
        return

    model, params = models[model_name]
    all_results = []

    # 使用网格搜索进行参数调整
    scoring = ['accuracy', 'precision','recall','f1']
    grid_search = GridSearchCV(model, params, cv=5, scoring=scoring,refit='f1',return_train_score=True,verbose=10,n_jobs=-1)
    for _ in tqdm(range(1), desc=f'ML model {model_name} is training...'):
        grid_search.fit(x_data, y_data)

    cv_results = grid_search.cv_results_

    for i in range(len(cv_results['params'])):
        result = {
            'index': i,
            'params': cv_results['params'][i],
            'mean_test_accuracy': cv_results['mean_test_accuracy'][i],
            'mean_test_precision': cv_results['mean_test_precision'][i],
            'mean_test_recall': cv_results['mean_test_recall'][i],
            'mean_test_f1': cv_results['mean_test_f1'][i],
        }
        all_results.append(result)

    # 输出网格搜索每组超参数的cv数据
    # for p, s in zip(grid_search.cv_results_['params'],
    #                 grid_search.cv_results_['mean_test_score']):
    #     print(p, s)
    #     all_results.append([p,s])

    # 将结果保存到文本文件
    with open(f'../data/splited_and_boosted_data/{dataset}/{dataset}_{model_name}_grid_search_results.txt', 'w') as file:
        for result in all_results:
            file.write(json.dumps(result) + '\n')

    # 网格搜索训练后的副产品(以F1-score为标准)
    print("模型的最优参数：", grid_search.best_params_)
    print("最优模型分数：", grid_search.best_score_)
    print("最优模型对象：", grid_search.best_estimator_)



def main():
    print("可选模型: SVM, Naive Bayes, Logistic Regression, Random Forest, Decision Tree")
    model_name = input("请输入要训练的模型名称: ")
    train_model(model_name)


if __name__ == "__main__":
    # Models and parameters
    models = {
        'SVM': (SVC(), [{'kernel': ['linear'], 'C': [2 ** x for x in range(-10, 11)]},
                        {'kernel': ['rbf', 'sigmoid'], 'C': [2 ** x for x in range(-10, 11)],
                         'gamma': [2 ** x for x in range(-10, 11)] + ['scale']},
                        {'kernel': ['poly'], 'C': [2 ** x for x in range(-10, 11)],
                         'gamma': [2 ** x for x in range(-10, 11)] + ['scale'], 'degree': [2, 3, 4, 5]}]),
        'Random Forest': (RandomForestClassifier(), {
            'max_features': [x/10 for x in range(1,11)] + ['sqrt','log2'],
            'max_samples': [x/10 for x in range(1,11)], 'n_jobs': [20]}),
        'Naive Bayes': (GaussianNB(), {'var_smoothing':[x/10 for x in range(1,11)]+[1e-09]}),
        'Logistic Regression': (LogisticRegression(), {'C': [x for x in range(1,11)]}),
        'Decision Tree': (DecisionTreeClassifier(), {'min_samples_split': [x for x in range(1,61)], 'min_samples_leaf':[x for x in range(1,61)]}),
        'test':(SVC(),{'kernel': ['rbf', 'sigmoid'], 'C': [1,2],'gamma':['scale',2]})
    }
    dataset = 'zookeeper'
    np.set_printoptions(threshold=np.inf)
    x_data,y_data = load_fold_data(0, 'all')
    main()

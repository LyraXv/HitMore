import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.preprocessing import MinMaxScaler
from cuml.svm import SVC as cuSVC
from cuml.naive_bayes import GaussianNB as cuGaussianNB
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
from cuml.preprocessing import StandardScaler as cuStandardScaler, MinMaxScaler as cuMinMaxScaler
import json
import cupy
from tqdm import tqdm
import warnings

from configx import ConfigX

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
        # Check the number of features in each dimension
        check_columns_num(br,cf,relat)
        # if br.shape[1] != 28: print("Attention: bugReportFeatures num!")
        # if cf.shape[1] != 77: print("Attention: buggyFileFeatures num!")
        # if relat.shape[1] != 18: print("Attention: RelationshhipFeatures num!")
        # data = merge_and_remove_duplicates([cf, relat])

        data = merge_and_remove_duplicates([br, cf, relat])
        # data = relat

        # dropNa
        data = data.dropna()
        fold_data.append(data)
        fold_indices.extend([i] * len(data))
    df_combined = pd.concat(fold_data)
    return df_combined,fold_indices


def load_fold_data():
    df_combined,fold_indices = readSplitedData()
    drop_columns = ['index', 'bugId', 'rank_0', 'score_0', 'path_0', 'rank_1', 'path_1', 'score_1', 'rank_2', 'path_2', 'label']
    data_info = df_combined[drop_columns]

    x_data = df_combined.drop(columns=drop_columns,axis=1)
    y_data = df_combined['label']

    print(f"当前训练数据量：{x_data.shape}")
    print(x_data.columns)

    # preprocess
    scaler = cuMinMaxScaler(feature_range=(0, 1))
    x_data = scaler.fit_transform(x_data)

    # Convert x_data and y_data to cupy arrays
    # x_data = cupy.asarray(x_data)  # Convert numpy array to cupy array
    # y_data = cupy.asarray(y_data)  # Convert numpy array to cupy array
    x_data = cupy.asarray(x_data,dtype=cupy.float32)  # Convert numpy array to cupy array
    y_data = cupy.asarray(y_data,dtype=cupy.int32)  # Convert numpy array to cupy array

    # 检查是否有非数值型数据
    print(f"Unique values in y_data: {np.unique(y_data.get())}")
    # 检查数据是否包含NaN
    print(f"NaN in x_data: {np.isnan(x_data.get()).any()}")
    print(f"NaN in y_data: {np.isnan(y_data.get()).any()}")

    return fold_indices, x_data , y_data , data_info

def calculateAvgMetrics(metricList):
    return sum(metricList)/len(metricList)

def search_bugCmit(bugId,dataset):
    path = f"../data/ordered_bugCmit/{dataset}"
    with open(path,'r')as f:
        commits = [line.strip().split(',') for line in f.readlines()]
        df_commits = pd.DataFrame(commits,columns=['bugId','cmit','date'])
    return df_commits[df_commits['bugId']==str(bugId)]['cmit'].values[0]

def reprocessedPath(path_2,path_1,path_0):
    if len(path_2)!=0:
        path = path_2.replace('/', '\\')
    elif len(path_1)!=0:
        pre_path = path_1.split('.java')[0]
        path = pre_path.replace('.','\\')
    elif len(path_0)!=0:
        pre_path = path_0.split('.java')[0]
        path = pre_path.replace('.','\\')
    else:
        path = ""
        print("Attention path is Nan")
    return path



def save_truly_buggy_file_info(y_pred,file_info,fold_idx):
    # y_pred,file_info 数据类型？
    positive_indices = np.where(y_pred == 1)[0]
    # 获取对应的 data_info 行
    positive_data_info = file_info.iloc[positive_indices]

    file_list = []
    #
    # # iterrow positive_data_info
    for index,row in positive_data_info.iterrows():
        index = str(fold_idx) +"_"+str(row['index'])
        bugId = row['bugId']
        filepath = reprocessedPath(row['path_2'],row['path_1'],row['path_0'])
        bugCmit = search_bugCmit(bugId,dataset)
        file = [index,bugId,filepath,bugCmit]
        file_list.append(file)
    if fold_idx==0:
        data_frame = pd.DataFrame(file_list,columns=['index','bugID', 'filePath','commit'])
        data_frame.to_csv(f"{dataset}_truly_buggy_file_result.csv", sep=',', mode='a', index=False)
    else:
        data_frame = pd.DataFrame(file_list)
        data_frame.to_csv(f"{dataset}_truly_buggy_file_result.csv", sep=',', mode='a', index=False,header=0)


def train_model(model_name):
    models_list = {'zookeeper':zookeeper_models,'openjpa':openjpa_models,'aspectj':aspectj_models,'Tomcat':tomcat_models,'hibernate':hibernate_models,'lucene':lucene_models}
    models  = models_list[dataset]
    if model_name not in models:
        print(f"模型 '{model_name}' 不存在，请选择以下模型之一: {list(models.keys())}")
        return

    model= models[model_name]
    ps = PredefinedSplit(test_fold=fold_indices)

    pipeline = Pipeline([
        ('smote', SMOTE(random_state=0)),
        ('model', model)
    ])

    Accuracy,Precision,Recall,F1 = [],[],[],[]
    AUC = []

    for fold_idx, (train_idx, test_idx) in enumerate(ps.split()):
        print(f"正在处理第 {fold_idx + 1} 折数据...")

        # 训练当前折数据
        pipeline.fit(x_data[train_idx].get(), y_data[train_idx].get())

        # 实时计算当前折的评估指标
        y_pred = pipeline.predict(x_data[test_idx].get())

        # save_truly_buggy_file_info(y_pred,data_info.iloc[test_idx],fold_idx)

        accuracy = pipeline.score(x_data[test_idx].get(), y_data[test_idx].get())
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_data[test_idx].get(), y_pred, average='weighted')
        auc = roc_auc_score(y_data[test_idx].get(), y_pred,average='micro')
        AUC.append(auc)
        Accuracy.append(accuracy)
        Precision.append(weighted_precision)
        Recall.append(weighted_recall)
        F1.append(weighted_f1)

    print([calculateAvgMetrics(Accuracy),calculateAvgMetrics(Precision),calculateAvgMetrics(Recall),calculateAvgMetrics(F1)])
    print(calculateAvgMetrics(AUC))



def main():
    model_name = 'Random Forest'
    print("Current model： ", model_name)
    train_model(model_name)


if __name__ == "__main__":
    zookeeper_models = {
        'Random Forest':cuRandomForestClassifier(max_features = 0.5,max_samples = 0.1),
        'SVM': cuSVC(C=4, gamma=1 / 64, kernel='rbf'),
        'Naive Bayes':cuGaussianNB(var_smoothing = 0.1),
        'Logistic Regression':cuLogisticRegression(C=7)
    }
    aspectj_models = {
        'Random Forest': cuRandomForestClassifier(max_features=0.2, max_samples=0.7),
        'SVM': cuSVC(C=1024, gamma=1 / 16, kernel='rbf'),
        'Naive Bayes': cuGaussianNB(var_smoothing=1e-09),
        'Logistic Regression': cuLogisticRegression(C=1)
    }
    tomcat_models = {
        'Random Forest': cuRandomForestClassifier(max_features=0.2, max_samples=0.4),
        'SVM': cuSVC(C=1, gamma=1 / 32, kernel='rbf'),
        'Naive Bayes': cuGaussianNB(var_smoothing=0.1),
        'Logistic Regression': cuLogisticRegression(C=1)

    }
    openjpa_models = {
        'Random Forest': cuRandomForestClassifier(max_features=0.4, max_samples=0.1),
        'SVM': cuSVC(C=128, gamma=1 / 512, kernel='rbf'),
        'Naive Bayes': cuGaussianNB(var_smoothing=0.2),
        'Logistic Regression': cuLogisticRegression(C=1)

    }
    hibernate_models = {
        'Random Forest': cuRandomForestClassifier(max_features=0.7, max_samples=0.3),
        'SVM': cuSVC(C=64, kernel='linear'),
        'Naive Bayes': cuGaussianNB(var_smoothing=0.1),
        'Logistic Regression': cuLogisticRegression(C=1)
    }
    lucene_models = {
        'Random Forest': cuRandomForestClassifier(max_features=0.8, max_samples=0.1),
        'SVM': cuSVC(C=256, gamma=1 / 512, kernel='rbf'),
        'Naive Bayes': cuGaussianNB(var_smoothing=0.1),
        'Logistic Regression': cuLogisticRegression(C=1)

    }


    configx = ConfigX()
    for dataset, file in configx.filepath_dict.items():
        # if dataset != 'zookeeper':
        #     continue
        print(f"===dataset:{dataset}===")
        np.set_printoptions(threshold=np.inf)
        fold_indices, x_data , y_data ,data_info = load_fold_data()
        main()

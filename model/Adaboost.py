import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.tree import DecisionTreeRegressor

from configx.configx import ConfigX
from utils.dataPartitioning import five_cross_validation, data_splited_by_time


def train_AdaBoost(train_x,train_y,test_x,test_y):
    base_estimator = DecisionTreeRegressor(max_depth=4)
    ada_regressor = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=50, random_state=42)
    ada_regressor.fit(train_x, train_y)
    predict_label = ada_regressor.predict(test_x)
    predict_label = [1 if x >= 0.5 else 0 for x in predict_label]

    accuracy = accuracy_score(test_y, predict_label)
    recall = recall_score(test_y, predict_label, average='weighted')
    f1 = f1_score(test_y, predict_label, average='weighted')
    precision = precision_score(test_y, predict_label, average='weighted')

    return accuracy,recall,f1,precision

if __name__ == '__main__':
    configx = ConfigX()
    Preprocess = False

    for dataset in configx.filepath_dict:
        if dataset in ['lucene']:
            continue
        print(f">>>>>Begin to train Adaboost for {dataset}<<<<<")
        # five cross validation
        folds = five_cross_validation(dataset,Preprocess)
        Accuracy =[]
        Recall =[]
        F1 = []
        Precision = []
        for i, (train_x, train_y, test_x, test_y) in enumerate(folds):
            accuracy,recall,f1,precision = train_AdaBoost(train_x,train_y,test_x,test_y)
            Accuracy.append(accuracy)
            Recall.append(recall)
            F1.append(f1)
            Precision.append(precision)
            print(f"{i}: acc {accuracy} recall {recall} f1 {f1} precision{precision}")

        with open("../data/get_info/Adaboost_note.txt", 'a') as f:
            f.write(
                f"Dataset:{dataset}  Preprocess:{Preprocess} Accuracy_Score: {format(sum(Accuracy)/5,'.4f')} Recall_Score: {format(sum(Recall)/5,'.4f')} F1-Score: {format(sum(F1)/5,'.4f')} Precision: {format(sum(Precision)/5,'.4f')}\n")

        print(f"Five cross validation finished!")

        # Validation by time
        fold = data_splited_by_time(dataset,Preprocess)
        train_x,train_y,test_x,test_y = fold
        accuracy,recall,f1,precision = train_AdaBoost(train_x,train_y,test_x,test_y)
        print(f"Time: acc {accuracy} recall {recall} f1 {f1} precision{precision}")

        with open("../data/get_info/Adaboost_note.txt", 'a') as f:
            f.write(
                f"Dataset(Time):{dataset}  Preprocess:{Preprocess} Accuracy_Score: {format(accuracy,'.4f')} Recall_Score: {format(recall,'.4f')} F1-Score: {format(f1,'.4f')} Precision: {format(precision,'.4f')}\n")
        print(f"Time validation finished!")








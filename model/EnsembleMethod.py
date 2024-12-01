import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.tree import DecisionTreeRegressor

from configx.configx import ConfigX
from utils.dataPartitioning import five_cross_validation, data_splited_by_time
from xgboost import XGBRegressor

def get_hit_k_mean(Hit):
    df_hit_k = pd.DataFrame(Hit)
    return df_hit_k.mean()

# 计算三个技术的hit@k
def other_approaches_hit(test_x,test_y):
    hits_k_amalgam = {}
    hits_k_bugLocator = {}
    hits_k_blizzard = {}

    test_df = pd.concat([test_x,test_y],axis=1)
    grouped = test_df.groupby('bugId')
    for k in [1,5,10,15,20]:
        hit_amalgam_list = []
        hit_bugLocator_list = []
        hit_blizzard_list = []
        bugid_list = []
        for bugid, group in grouped:
           bugid_list.append(bugid)
           if ((group['label']==1) & (group['rank_0']<k)).any():
               hit_amalgam = 1
           else:
               hit_amalgam = 0
           if ((group['label'] == 1) & (group['rank_1']<k)).any():
               hit_bugLocator = 1
           else:
               hit_bugLocator = 0
           if ((group['label'] == 1) & (group['rank_2']<k)).any():
               hit_blizzard = 1
           else:
               hit_blizzard = 0
           hit_amalgam_list.append(hit_amalgam)
           hit_bugLocator_list.append(hit_bugLocator)
           hit_blizzard_list.append(hit_blizzard)

        hits_k_amalgam[f"Hit@{k}"] = sum(hit_amalgam_list)/len(hit_amalgam_list)
        hits_k_bugLocator[f"Hit@{k}"] = sum(hit_bugLocator_list)/len(hit_bugLocator_list)
        hits_k_blizzard[f"Hit@{k}"] = sum(hit_blizzard_list)/len(hit_blizzard_list)
        # print(f"K:{k}sum:{sum(hit_amalgam_list)} Len: {len(hit_amalgam_list)}")
    return hits_k_amalgam,hits_k_bugLocator,hits_k_blizzard

def accuracy_at_k(pred_results,k):
    print(f">>>>>>k::{k}<<<<<<<<<")
    accuracy_list = []
    hit_list = []

    grouped = pred_results.groupby('bugId')
    for bugid, group in grouped:
        sorted_group = group.sort_values(by='pred_score', ascending=False)
        if len(sorted_group) < k:
            top_k = sorted_group.copy()
        else:
            top_k = sorted_group.head(k).copy()
        top_k['pred_label'] = (top_k['pred_score'] >= 0.5).astype(int)
        acc = accuracy_score(top_k['true_label'], top_k['pred_label'])
        accuracy_list.append(acc)
        # print("===",bugid,"===")
        # print(top_k)

        if (((top_k['true_label'] == 1) ).any()):
            hit = 1
        else:
            hit = 0
        # print(bugid)
        # print(top_k)
        hit_list.append(hit)
    print(f"sum:{sum(hit_list)},len:{len(hit_list)}")
    return sum(accuracy_list) / len(accuracy_list),sum(hit_list)/len(hit_list)

# fold(i) 's top20 files
def save_top20_i(dataset,i,pred_results):
    # read mergedData
    df_merged = pd.read_csv('../data/get_info/' + dataset + '/mergedRecList.csv')
    df_merged = df_merged.rename(columns={'Unnamed: 0':'id'})
    df_result = pd.DataFrame(columns=['id','bugId','rank_0','score_0','path_0','rank_1','path_1','score_1','rank_2','path_2','label'])

    grouped = pred_results.groupby('bugId')
    for bugid, group in grouped:
        sorted_group = group.sort_values(by='pred_score', ascending=False)
        if len(sorted_group) < 20:
            top_k = sorted_group.copy()
        else:
            top_k = sorted_group.head(20).copy()
        temp_df = pd.merge(df_merged, top_k[['id']], on='id', how='inner') # only save data in df_merged
        df_result = pd.concat([df_result, temp_df], ignore_index=True)

    # save as csv
    df_result = df_result.drop(columns='id')
    df_result.to_csv(f"../data/splited_and_boosted_data/{dataset}/{i}.csv")

def save_all(dataset,i,pred_results):
    # read mergedData
    df_merged = pd.read_csv('../data/get_info/' + dataset + '/mergedRecList.csv')
    df_merged = df_merged.rename(columns={'Unnamed: 0':'id'})
    df_result = pd.DataFrame(columns=['id','bugId','rank_0','score_0','path_0','rank_1','path_1','score_1','rank_2','path_2','label'])

    grouped = pred_results.groupby('bugId')
    for bugid, group in grouped:
        sorted_group = group.sort_values(by='pred_score', ascending=False)
        print(sorted_group)
        temp_df = pd.merge(sorted_group[['id']], df_merged, on='id', how='inner')
        # temp_df = pd.merge(df_merged, sorted_group[['id']], on='id', how='inner') # only save data in df_merged
        # print(temp_df)
        # exit()
        df_result = pd.concat([df_result, temp_df], ignore_index=True)

    # save as csv
    df_result = df_result.drop(columns='id')
    df_result.to_csv(f"../data/hitmore_initial_rec_list/{dataset}/{i}.csv")

def train_model(train_x,train_y,test_x,test_y,model_name):
    if model_name == 'Adaboost':
        base_estimator = DecisionTreeRegressor(max_depth=4)
        model = AdaBoostRegressor(base_estimator=base_estimator, learning_rate=0.01,n_estimators=50, random_state=42)
    elif model_name == 'XGboost': # max_depth=3
        model = XGBRegressor(max_depth=3,learning_rate=0.01,min_child_weight=5,colsample_bytree=1.0,subsample=0.6,gamma=0)

    model.fit(train_x.iloc[:,2:], train_y)
    predict_label = model.predict(test_x.iloc[:,2:])

    # Accuracy@k
    accuracies = {}
    hits = {}
    pred_results = pd.DataFrame({'id':test_x.iloc[:,0],'bugId':test_x.iloc[:,1],'true_label': test_y, 'pred_score': predict_label})
    for k in [1,5,10,15,20]:
        accuracies[f"Accuracy@{k}"],hits[f"Hit@{k}"] = accuracy_at_k(pred_results,k)
    if configx.saveTop20RecLists:
        save_top20_i(dataset,i,pred_results)
    if configx.saveInitialRecLists:
        save_all(dataset,i,pred_results)

    predict_label = [1 if x >= 0.5 else 0 for x in predict_label]

    # accuracy = accuracy_score(test_y, predict_label)
    recall = recall_score(test_y, predict_label, average='weighted')
    f1 = f1_score(test_y, predict_label, average='weighted')
    precision = precision_score(test_y, predict_label, average='weighted')

    return accuracies,hits,recall,f1,precision

if __name__ == '__main__':
    configx = ConfigX()
    Preprocess = configx.preprocess

    # setView
    pd.set_option('display.expand_frame_repr', False)

    # other approaches
    hit_approachs={'amalgam':0,'bugLocator':0,'blizzard':0}

    for dataset in configx.filepath_dict:
        # if dataset not in ['Tomcat']:
        #     continue
        print(f">>>>>Begin to train Adaboost for {dataset}<<<<<")
        # five cross validation
        folds = five_cross_validation(dataset,configx,Preprocess)
        Accuracy =[]
        Recall =[]
        F1 = []
        Precision = []
        Hit,Hit_amalgam,Hit_bugLocator,Hit_blizzard = [],[],[],[]
        for i, (train_x, train_y, test_x, test_y) in enumerate(folds):
            accuracy,hit,recall,f1,precision = train_model(train_x,train_y,test_x,test_y,configx.RecModel)
            Accuracy.append(accuracy)
            Recall.append(recall)
            F1.append(f1)
            Precision.append(precision)
            Hit.append(hit)
            print(f"{i}: acc {accuracy} recall {recall} f1 {f1} precision{precision}\nhit:{hit}")
            # other approaches hit
            # hit_ama,hit_loc,hit_bli = other_approaches_hit(test_x,test_y)
            # Hit_amalgam.append(hit_ama)
            # Hit_bugLocator.append(hit_loc)
            # Hit_blizzard.append(hit_bli)

        # print(f"OtherApproachesHits:\nAmalgam: {get_hit_k_mean(Hit_amalgam)}\nBugLocator: {get_hit_k_mean(Hit_bugLocator)}\n"
        #       f"Blizzard: {get_hit_k_mean(Hit_blizzard)}")
        # Accuracy@k
        accuracy_k = get_hit_k_mean(Accuracy)
        # Hit@k
        hit_k = get_hit_k_mean(Hit)

        with open("../data/get_info/Adaboost_note.txt", 'a') as f:
            f.write(f">>>>>>RawDataType:{configx.rawDataType} Model:{configx.RecModel}<<<<<<\n")
            f.write(
                f"Dataset:{dataset}  Preprocess:{Preprocess} \nAccuracy_Score: \n{accuracy_k} \nHit@k:\n{hit_k}\nRecall_Score: {format(sum(Recall)/5,'.4f')} F1-Score: {format(sum(F1)/5,'.4f')} Precision: {format(sum(Precision)/5,'.4f')}\n")

        print(f"Five cross validation finished!")

        # Validation by time
        # i = 5
        # fold = data_splited_by_time(dataset,configx,Preprocess)
        # train_x,train_y,test_x,test_y = fold
        # accuracy_k,hit_k,recall,f1,precision = train_model(train_x,train_y,test_x,test_y,configx.RecModel)
        # print(f"Time: acc {accuracy} recall {recall} f1 {f1} precision{precision}")
        #
        # with open("../data/get_info/Adaboost_note.txt", 'a') as f:
        #     f.write(
        #         f"Dataset(Time):{dataset}  Preprocess:{Preprocess} Accuracy_Score: {accuracy_k} Recall_Score: {format(recall,'.4f')} F1-Score: {format(f1,'.4f')} Precision: {format(precision,'.4f')}\n Hit{hit_k}\n")
        # print(f"Time validation finished!")


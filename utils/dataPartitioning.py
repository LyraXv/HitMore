import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.utils import readBugIdByTime


def five_cross_validation(dataset,preprocess=False):
    # 读取CSV文件
    data = pd.read_csv('../data/get_info/'+ dataset + '/recommendedList2.csv')
    data = data.drop(columns=['SourceFile'])

    if preprocess:
        data = preprocessRecListsData(data)

    # 创建KFold对象，设置n_splits为5（五折）
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 创建一个空列表用于存储划分结果
    folds = []

    # 按照BugId进行分组
    grouped = data.groupby('BugId')

    # 创建一个包含所有BugId的列表
    bug_ids = list(grouped.groups.keys())

    # 使用KFold对象进行五折划分
    index = 0
    for train_index, test_index in kf.split(bug_ids):
        # 获取训练集和测试集的BugId
        train_bug_ids = [bug_ids[i] for i in train_index]
        test_bug_ids = [bug_ids[i] for i in test_index]

        # train_bug_ids.to_csv(f'../data/get_info/{dataset}/dataPartion/train_bugid_fold_{index + 1}.csv', index=False)
        # test_bug_ids.to_csv(f'../data/get_info/{dataset}/dataPartion/test_bugid_fold_{index + 1}.csv', index=False)

        # 获取训练集和测试集的数据
        train_data = data[data['BugId'].isin(train_bug_ids)]
        test_data = data[data['BugId'].isin(test_bug_ids)]

        # 将数据分为特征和目标
        train_x = train_data.iloc[:, :-1]  # 除最后一列之外的所有列
        train_y = train_data.iloc[:, -1]  # 最后一列
        test_x = test_data.iloc[:, :-1]  # 除最后一列之外的所有列
        test_y = test_data.iloc[:, -1]  # 最后一列

        # 将结果添加到folds列表
        folds.append((train_x, train_y, test_x, test_y))
    return  folds
    # 打印每折的训练集和测试集
    # for i, (train_x, train_y, test_x, test_y) in enumerate(folds):
    #     print(f'Fold {i + 1}:')
    #     print('Train set:')
    #     print(train_x)
    #     print(train_y)
    #     print('Test set:')
    #     print(test_x)
    #     print(test_y)
    #     print()

    # 将每折的数据保存到文件中，使用以下代码
    # for i, (train_x, train_y, test_x, test_y) in enumerate(folds):
    #     train_x.to_csv(f'train_x_fold_{i + 1}.csv', index=False)
    #     train_y.to_csv(f'train_y_fold_{i + 1}.csv', index=False)
    #     test_x.to_csv(f'test_x_fold_{i + 1}.csv', index=False)
    #     test_y.to_csv(f'test_y_fold_{i + 1}.csv', index=False)


def data_splited_by_time(dataset,preprocess=False):
    bugId = readBugIdByTime(dataset)
    split_index = int(len(bugId)*0.8)

    data = pd.read_csv('../data/get_info/' + dataset + '/recommendedList2.csv')
    # print(bugId)
    data = data.drop(columns=['SourceFile'])
    if preprocess:
        data = preprocessRecListsData(data)

    train_id = bugId.iloc[:split_index]
    test_id = bugId.iloc[split_index:]

    train_bug_ids = train_id.tolist()
    test_bug_ids = test_id.tolist()

    train_data = data[data['BugId'].isin(train_bug_ids)]
    test_data =  data[data['BugId'].isin(test_bug_ids)]

    if train_data.empty:
        print("Train data is empty!")
    if test_data.empty:
        print("Test data is empty!")


    train_x = train_data.iloc[:, :-1]
    train_y = train_data.iloc[:, -1]
    test_x = test_data.iloc[:, :-1]
    test_y = test_data.iloc[:, -1]

    return (train_x,train_y,test_x,test_y)

def preprocessRecListsData(data):
    # 初始化Min-Max Scaler
    # scaler = MinMaxScaler()
    # Zcore标准化
    scaler = StandardScaler()

    # 对数据进行归一化
    data[['Rank_0','Score_0', 'Rank_1','Score_1', 'Rank_2']] = scaler.fit_transform(data[['Rank_0','Score_0', 'Rank_1','Score_1', 'Rank_2']])

    return data



if __name__ == '__main__':
    five_cross_validation()
    # data_splited_by_time()
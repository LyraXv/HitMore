import pandas as pd

# data = pd.read_csv('D:\\PycharmProjects\\hitMore\\data\\zookeeper_dat\\res.csv')

data = pd.read_csv('/data\\zookeeper_dat\\res.csv')
data = data.fillna(
    0,  # nan的替换值
    inplace=False  # 是否跟换源文件
)
data = data.drop(axis=1, columns=['Unnamed: 0', 'SourceFile', 'SourceFile_txt', 'fixedCmit', 'Summary', 'Description',
                                  'Pre_bug_text', 'file', 'class',
                                  'totalMethodsQty', 'staticMethodsQty', 'publicMethodsQty', 'privateMethodsQty',
                                  'protectedMethodsQty',
                                  'defaultMethodsQty', 'visibleMethodsQty', 'abstractMethodsQty', 'finalMethodsQty',
                                  'synchronizedMethodsQty',
                                  'totalFieldsQty', 'staticFieldsQty', 'publicFieldsQty', 'privateFieldsQty',
                                  'protectedFieldsQty',
                                  'defaultFieldsQty', 'finalFieldsQty', 'synchronizedFieldsQty', 'returnQty', 'loopQty',
                                  'comparisonsQty',
                                  'tryCatchQty', 'parenthesizedExpsQty', 'stringLiteralsQty', 'numbersQty',
                                  'assignmentsQty',
                                  'mathOperationsQty', 'modifiers', 'logStatementsQty'])
data

readability = pd.read_csv('./Readability/zookeeper.csv')
readability = readability.drop(axis=1, columns=['Unnamed: 0'])
data = data.merge(readability, on=['BugId'])
data

reporters = pd.read_csv('./Reporters/zookeeper.csv')
reporters = reporters.drop(axis=1, columns=['Unnamed: 0', 'Reporter', 'Assignee', 'create_time', 'resolve_time'])
data = data.merge(reporters, on=['BugId'])
data

# 切分数据集 按BugId切分 80%训练 20%测试
BugList = data['BugId'].drop_duplicates()
# train_bug_ids = BugList.sample(frac=0.8, random_state=666)
# train_bug_ids
BugList = BugList.values.tolist()

# 五则交叉
from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits=5, random_state=42, shuffle=True)
train_bug_id = []  # 存放五则的训练集bug_id划分
test_bug_id = []  # 存放五则的测试集bug_id划分
for k, (train_bug_index, test_bug_index) in enumerate(kf.split(BugList)):
    #     print('train_index: %s, test_index: %s' %(train_bug_index, test_bug_index))
    train_bug_id.append(np.array(BugList)[train_bug_index])
    test_bug_id.append(np.array(BugList)[test_bug_index])

# 从sklearn中导入决策树模型
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
import sklearn.preprocessing as preproc
from imblearn.over_sampling import SMOTE

total_train_accuracy = 0
total_test_accuracy = 0

for i in range(5):
    print("Round ", i)
    train_data = data[data.BugId.isin(train_bug_id[i])]
    test_data = data[data.BugId.isin(test_bug_id[i])]

    data_train = train_data.drop(axis=1, columns=['Label', 'BugId'])
    tag_train = train_data['Label']
    data_test = test_data.drop(axis=1, columns=['Label', 'BugId'])
    tag_test = test_data['Label']

    # 特征缩放至 0-1 切分前！
    data_train = preproc.normalize(data_train)
    data_test = preproc.normalize(data_test)

    # 过采样
    oversampler = SMOTE(random_state=0)
    os_features, os_labels = oversampler.fit_resample(data_train, tag_train)
    print(os_features)
    print(os_labels)

    # 定义决策树模型 
    clf = DecisionTreeClassifier(criterion='entropy')

    # 在训练集上训练决策树模型
    #     clf.fit(data_train, tag_train)
    clf.fit(os_features, os_labels)

    # 在训练集和测试集上利用训练好的模型进行预测
    train_predict = clf.predict(data_train)
    test_predict = clf.predict(data_test)

    # 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
    print('The accuracy of the train_DecisionTree is:', metrics.accuracy_score(tag_train, train_predict))
    print('The accuracy of the test_DecisionTree is:', metrics.accuracy_score(tag_test, test_predict))

    P = 0
    TP = 0
    FN = 0
    # “1”类的precision、recall值
    test = tag_test.values.tolist()
    for i in range(len(test_predict)):
        #         print(test_predict[i], test[i])
        if (test_predict[i] == 1):
            P += 1
        if (test_predict[i] == test[i] == 1):
            TP += 1
        if (test_predict[i] == 0 and test[i] == 1):
            FN += 1
    precision = TP / P
    recall = TP / (TP + FN)
    print('precision: ', precision)
    print('recall: ', recall)

    total_train_accuracy += metrics.accuracy_score(tag_train, train_predict)
    total_test_accuracy += metrics.accuracy_score(tag_test, test_predict)

# 五则取平均
mean_train_accuracy = total_train_accuracy / 5
mean_test_accuracy = total_test_accuracy / 5
print("//////////////////////////////////////////////")
print("Average train accuracy: ", mean_train_accuracy)
print("Average test accuracy: ", mean_test_accuracy)


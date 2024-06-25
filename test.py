# import pandas as pd
#
# bl_openjpa = pd.read_csv('data/openjpa_dat/buglocator_openjpa.csv')
#
# openjpa_bugids = bl_openjpa['BugId']
# openjpa_sourcefiles = bl_openjpa['SourceFile']
#
# openjpa_sourcefilesFeatures = pd.read_csv('data/openjpa_dat/openjpa_dataclass.csv')
#
# # for bugid in openjpa_bugids:
# #     print(bugid)
#
# for sourcefiles in openjpa_sourcefiles:
#     print(sourcefiles)
#     for sf in openjpa_sourcefilesFeatures['class']:
#         if(sourcefiles.strip('.java') == sf):
#             print(sourcefiles)
#             bl_openjpa.append()

import pandas as pd
import numpy as np
data = pd.read_csv('C:\\Users\\BlueWeirdo\\PycharmProjects\\hitMore\\data\\zookeeper_dat\\res_final.csv')
data = data.drop(axis=1, columns=['Unnamed: 0', 'Patch', 'Screenshots', 'Itemization', 'StackTrace', 'Keywords'])

br = pd.read_csv('D:\\zookeeper\\br_features_1.csv')
br = br.drop(axis=1, columns=['Summary', 'Description', 'Pre_bug_text'])
data = data.merge(br, on=['BugId'])

reporters = pd.read_csv('D:\\Reporters\\zookeeper.csv')
reporters = reporters.drop(axis=1, columns=['Unnamed: 0', 'Reporter', 'Assignee', 'create_time', 'resolve_time'])
data = data.merge(reporters, on=['BugId'])

readability = pd.read_csv('D:\\Readability\\zookeeper.csv')
readability = readability.drop(axis=1, columns=['Unnamed: 0'])
data = data.merge(readability, on=['BugId'])

similarity = pd.read_csv('D:\\features_all.csv').rename(columns={'report_id': 'BugId', 'file': 'SourceFile'})
similarity = similarity.drop_duplicates()
data = data.merge(similarity, on=['BugId', 'SourceFile'])

print(data)
# res = res.drop(axis=1, columns=['Unnamed: 0'])
# label1 = pd.read_csv('D:\\PycharmProjects\\hitMore\\data\\zookeeper_dat\\res_label1.csv').rename(columns={'label': 'Label'})
# label1 = label1.drop(axis=1, columns=['Unnamed: 0'])
# data = pd.read_csv('data/tomcat_dat/test.csv')
# data = pd.read_csv('data/openjpa_dat/test.csv')
# res = pd.read_csv('D:\\res.csv')
# label1 = pd.read_csv('D:\\res_label1.csv').rename(columns={'label': 'Label'})
# data = pd.concat([res, label1]).drop_duplicates()
# data.to_csv('D:\\res_zookeeper.csv')
# data = pd.read_csv('D:\\tomcat\\res.csv')

# data = data.drop(axis=1, columns=['Unnamed: 0', 'SourceFile', 'SourceFile_txt', 'fixedCmit', 'Summary', 'Description', 'Pre_bug_text', 'Pre_BugReport', 'file', 'class',
#                                  'totalMethodsQty', 'staticMethodsQty', 'publicMethodsQty', 'privateMethodsQty', 'protectedMethodsQty',
#                                   'defaultMethodsQty', 'visibleMethodsQty', 'abstractMethodsQty', 'finalMethodsQty', 'synchronizedMethodsQty',
#                                   'totalFieldsQty', 'staticFieldsQty', 'publicFieldsQty', 'privateFieldsQty', 'protectedFieldsQty',
#                                   'defaultFieldsQty', 'finalFieldsQty', 'synchronizedFieldsQty', 'returnQty', 'loopQty', 'comparisonsQty',
#                                   'tryCatchQty', 'parenthesizedExpsQty', 'stringLiteralsQty', 'numbersQty', 'assignmentsQty',
#                                   'mathOperationsQty', 'modifiers', 'logStatementsQty'])
# data_label1 = pd.read_csv('D:\\PycharmProjects\\hitMore\\data\\zookeeper_dat\\res_label1.csv').rename(columns={'label': 'Label'})
# data_label1 = data_label1.fillna(
#     0,  # nan的替换值
#     inplace=False  # 是否跟换源文件
# )
# data_label1 = data_label1.drop(axis=1, columns=['Unnamed: 0', 'SourceFile', 'SourceFile_txt', 'fixedCmit', 'Summary', 'Description', 'Pre_bug_text', 'file', 'class',
#                                  'totalMethodsQty', 'staticMethodsQty', 'publicMethodsQty', 'privateMethodsQty', 'protectedMethodsQty',
#                                   'defaultMethodsQty', 'visibleMethodsQty', 'abstractMethodsQty', 'finalMethodsQty', 'synchronizedMethodsQty',
#                                   'totalFieldsQty', 'staticFieldsQty', 'publicFieldsQty', 'privateFieldsQty', 'protectedFieldsQty',
#                                   'defaultFieldsQty', 'finalFieldsQty', 'synchronizedFieldsQty', 'returnQty', 'loopQty', 'comparisonsQty',
#                                   'tryCatchQty', 'parenthesizedExpsQty', 'stringLiteralsQty', 'numbersQty', 'assignmentsQty',
#                                   'mathOperationsQty', 'modifiers', 'logStatementsQty'])
# # data_label1 = data_label1.drop(axis=1, columns=['Unnamed: 0', 'SourceFile', 'SourceFile_txt', 'fixedCmit', 'Summary', 'Description', 'Pre_bug_text', 'file', 'class'])
#
# data = pd.concat([data_label1, data], axis=0)
# data = data.drop_duplicates()
# print(data)

# data = data.drop(axis=1, columns=['Unnamed: 0', 'Unnamed: 0.1', 'SourceFile', 'SourceFile_txt', 'fixedCmit', 'Summary',
#                                   'Description', 'Pre_bug_text', 'file', 'class',
                                  # 'totalMethodsQty', 'staticMethodsQty', 'publicMethodsQty', 'privateMethodsQty',
                                  # 'protectedMethodsQty', 'defaultMethodsQty', 'visibleMethodsQty', 'abstractMethodsQty',
                                  # 'finalMethodsQty', 'synchronizedMethodsQty', 'totalFieldsQty', 'staticFieldsQty',
                                  # 'publicFieldsQty', 'privateFieldsQty', 'protectedFieldsQty', 'defaultFieldsQty',
                                  # 'finalFieldsQty', 'synchronizedFieldsQty', 'nosi', 'returnQty', 'loopQty', 'comparisonsQty',
                                  # 'tryCatchQty', 'parenthesizedExpsQty', 'stringLiteralsQty', 'numbersQty',
                                  # 'assignmentsQty', 'mathOperationsQty', 'variablesQty', 'maxNestedBlocksQty', 'anonymousClassesQty',
                                  # 'innerClassesQty', 'lambdasQty', 'uniqueWordsQty', 'modifiers', 'logStatementsQty', 'lcc'
                                  # ])

# print(data)

# readability = pd.read_csv('../../Dataset/Readability/zookeeper.csv')
# readability = readability.drop(axis=1, columns=['Unnamed: 0'])
# data = data.merge(readability, on=['BugId'])
# print(data)

# reporters = pd.read_csv('../../Dataset/Reporters/zookeeper.csv')
# reporters = reporters.drop(axis=1, columns=['Unnamed: 0', 'Reporter', 'Assignee', 'create_time', 'resolve_time'])
# data = data.merge(reporters, on=['BugId'])

# similarity = pd.read_csv('data/zookeeper_dat/features_all.csv').rename(columns={'report_id': 'BugId', 'file': 'SourceFile'})
# similarity = similarity.drop_duplicates()
# data = data.merge(similarity, on=['BugId', 'SourceFile'])

# data = data.drop(axis=1, columns=['SourceFile', 'SourceFile_txt', 'fixedCmit', 'Summary',
#                                   'Description', 'Pre_bug_text', 'file', 'class',
#                                   'totalMethodsQty', 'staticMethodsQty', 'publicMethodsQty', 'privateMethodsQty',
#                                   'protectedMethodsQty', 'defaultMethodsQty', 'visibleMethodsQty', 'abstractMethodsQty',
#                                   'finalMethodsQty', 'synchronizedMethodsQty', 'totalFieldsQty', 'staticFieldsQty',
#                                   'publicFieldsQty', 'privateFieldsQty', 'protectedFieldsQty', 'defaultFieldsQty',
#                                   'finalFieldsQty', 'synchronizedFieldsQty', 'nosi', 'returnQty', 'loopQty', 'comparisonsQty',
#                                   'tryCatchQty', 'parenthesizedExpsQty', 'stringLiteralsQty', 'numbersQty',
#                                   'assignmentsQty', 'mathOperationsQty', 'variablesQty', 'maxNestedBlocksQty', 'anonymousClassesQty',
#                                   'innerClassesQty', 'lambdasQty', 'uniqueWordsQty', 'modifiers', 'logStatementsQty', 'lcc'
#                                   ])
data = data.drop(axis=1, columns=['SourceFile', 'SourceFile_txt', 'fixedCmit', 'Summary', 'Description', 'Pre_bug_text', 'Pre_BugReport', 'file', 'class'])
data = data.fillna(
    0,  # nan的替换值
    inplace=False  # 是否跟换源文件
)
print(data)
# print(np.isnan(data).any())
# print(np.isinf(data).all())
train_inf = np.isinf(data)
data[train_inf] = 0

# train_null = pd.isnull(data)
# train_null = data[train_null == True]
# print(train_null)

# 切分数据集 按BugId切分 80%训练 20%测试
BugList = data['BugId'].drop_duplicates()
# train_bug_ids = BugList.sample(frac=0.8, random_state=666)
# train_bug_ids
BugList = BugList.values.tolist()

# 五则交叉
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=42, shuffle=True)
train_bug_id = []   # 存放五则的训练集bug_id划分
test_bug_id = []   # 存放五则的测试集bug_id划分
for k, (train_bug_index, test_bug_index) in enumerate(kf.split(BugList)):
    # print('train_index: %s, test_index: %s' %(train_bug_index, test_bug_index))
    train_bug_id.append(np.array(BugList)[train_bug_index])
    test_bug_id.append(np.array(BugList)[test_bug_index])


# 评估回归性能
# criterion ：
# 回归树衡量分枝质量的指标，支持的标准有三种：
# 1）输入"mse"使用均方误差mean squared error(MSE)，父节点和叶子节点之间的均方误差的差额将被用来作为特征选择的标准，
# 这种方法通过使用叶子节点的均值来最小化L2损失
# 2）输入“friedman_mse”使用费尔德曼均方误差，这种指标使用弗里德曼针对潜在分枝中的问题改进后的均方误差
# 3）输入"mae"使用绝对平均误差MAE（mean absolute error），这种指标使用叶节点的中值来最小化L1损失

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import sklearn.preprocessing as preproc
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler


total_train_accuracy = 0
total_test_accuracy = 0
precision_0 = 0
recall_0 = 0
precision_1 = 0
recall_1 = 0


# 特征缩放
# print(data['classname_similarity'].max())
# print(data['classname_similarity'].min())
# class_sim = (data['classname_similarity'] - data['classname_similarity'].min()) / (data['classname_similarity'].max() - data['classname_similarity'].min())
# print(class_sim)
# bug_recency = (data['bug_recency'] - data['bug_recency'].min()) / (data['bug_recency'].max() - data['bug_recency'].min())
# bug_frequency = (data['bug_frequency'] - data['bug_frequency'].min()) / (data['bug_frequency'].max() - data['bug_frequency'].min())
# rd_ARI = (data['rd.ARI'] -;data['rd.ARI'].min()) / (data['rd.ARI'].max() - data['rd.ARI'].min())

# scaler = MinMaxScaler()
# scaler.fit(data[data.columns[2:73]])
# data = scaler.transform(data[data.columns[2:73]])
# print(data)

# data = preproc.normalize(data['cosine_sim', 'collab_filter', 'classname_similarity', 'bug_recency', 'bug_frequency',
#                               'rd.ARI', 'rd.FleschReadingEase', 'rd.FleschKincaidGradeLevel','rd.GunningFogIndex',
#                               'rd.SMOGIndex', 'rd.ColemanLiauIndex', 'rd.LIX', 'rd.RIX',
#                               'cbo', 'cboModified', 'fanin', 'fanout', 'wmc', 'dit', 'noc', 'rfc', 'lcom*', 'loc',
#                               'Patch', 'Screenshots', 'Itemization', 'StackTrace', 'Keywords',
#                               'commit_num', 'assign_num', 'valid_commit_num'])

for i in range(5):
    print("Round ", i)
    train_data = data[data.BugId.isin(train_bug_id[i])]
    test_data = data[data.BugId.isin(test_bug_id[i])]

    data_train = train_data.drop(axis=1, columns=['Label', 'BugId'])
    tag_train = train_data['Label']
    data_test = test_data.drop(axis=1, columns=['Label', 'BugId'])
    tag_test = test_data['Label']
    data_train = pd.DataFrame(train_data, columns=[ 'collab_filter', 'classname_similarity', 'cosine_sim', 'semantic_sim'])
    data_test = pd.DataFrame(test_data, columns=['collab_filter', 'classname_similarity', 'cosine_sim', 'semantic_sim'])
    #                               # 'rd.ARI', 'rd.FleschReadingEase', 'rd.FleschKincaidGradeLevel','rd.Gunnin
    #                               gFogIndex',
    #                               # 'rd.SMOGIndex', 'rd.ColemanLiauIndex', 'rd.LIX', 'rd.RIX',
    #                               # 'cbo', 'cboModified', 'fanin', 'fanout', 'wmc', 'dit', 'noc', 'rfc', 'lcom*', 'loc',
    #                               # 'Patch', 'Screenshots', 'Itemization', 'StackTrace', 'Keywords',
    #                               # 'commit_num', 'assign_num', 'valid_commit_num'
    #                                                ])
    # data_test = pd.DataFrame(test_data, columns=['collab_filter', 'classname_similarity', 'bug_recency', 'bug_frequency',
    #                               # 'rd.ARI', 'rd.FleschReadingEase', 'rd.FleschKincaidGradeLevel','rd.GunningFogIndex',
    #                               # 'rd.SMOGIndex', 'rd.ColemanLiauIndex', 'rd.LIX', 'rd.RIX',
    #                               # 'cbo', 'cboModified', 'fanin', 'fanout', 'wmc', 'dit', 'noc', 'rfc', 'lcom*', 'loc',
    #                               # 'Patch', 'Screenshots', 'Itemization', 'StackTrace', 'Keywords',
    #                               # 'commit_num', 'assign_num', 'valid_commit_num'
    #                                              ])

    # 特征缩放至 0-1
    # data_train = preproc.normalize(data_train)
    # data_test = preproc.normalize(data_train)
    scaler = MinMaxScaler()
    scaler.fit(data_train)
    scaler.data_max_
    data_train = scaler.transform(data_train)
    scaler.fit(data_test)
    scaler.data_max_
    data_test = scaler.transform(data_test)


    # 过采样
    oversampler = SMOTE(random_state=0)
    os_features, os_labels = oversampler.fit_resample(data_train, tag_train)
    # print(os_features)
    # print(os_labels)

    # 欠采样 随机欠采样
    rus = RandomUnderSampler(random_state=0)
    X_resampled_u1, y_resampled_u1 = rus.fit_resample(data_train, tag_train)
    # counter_resampled_u1 = Counter(y_resampled_u1)
    # print(counter_resampled_u1)

    forest = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                           class_weight='balanced', criterion='entropy', max_depth=6,
                           max_features=0.7602967141008143, max_leaf_nodes=None,
                           max_samples=None, min_impurity_decrease=1.922617919965002e-09, min_samples_leaf=5,
                           min_samples_split=5, min_weight_fraction_leaf=0.0,
                           n_estimators=28, n_jobs=-1, oob_score=False,
                           random_state=2934, verbose=0, warm_start=False)  # 实例化模型RandomForestClassifier

    # forest = RandomForestRegressor(n_estimators=1000,
    #                                criterion='squared_error',
    #                                random_state=1,
    #                                n_jobs=-1)

    # forest.fit(data_train, tag_train)
    # forest.fit(os_features, os_labels)
    forest.fit(X_resampled_u1, y_resampled_u1)

    y_train_pred = forest.predict(data_train)
    y_test_pred = forest.predict(data_test)
    # print(y_test_pred)

    for i in range(len(y_test_pred)):
        if y_test_pred[i] >= 0.5:
            y_test_pred[i] = 1
        else:
            y_test_pred[i] = 0

    for i in range(len(y_train_pred)):
        if y_train_pred[i] >= 0.5:
            y_train_pred[i] = 1
        else:
            y_train_pred[i] = 0

    # print(y_train_pred)
    # print(y_test_pred)

    # print('The accuracy of the train_RandomForest is:',
    #       metrics.accuracy_score(tag_train, y_train_pred))
    # print('The accuracy of the test_RandomForest is:',
    #       metrics.accuracy_score(tag_test, y_test_pred))

    total_train_accuracy += metrics.accuracy_score(tag_train, y_train_pred)
    total_test_accuracy += metrics.accuracy_score(tag_test, y_test_pred)

    P = 0
    TP = 0
    FN = 0
    # “0”类的precision、recall值
    test = tag_test.values.tolist()
    for i in range(len(y_test_pred)):
        if (y_test_pred[i] == 0):
            P += 1
        if (y_test_pred[i] == test[i] == 0):
            TP += 1
        if (y_test_pred[i] == 1 and test[i] == 0):
            FN += 1
    precision = TP / P
    recall = TP / (TP + FN)
    print('0-precision: ', precision)
    print('0-recall: ', recall)
    precision_0 += precision
    recall_0 += recall


    auc = metrics.roc_auc_score(tag_test, y_test_pred)  # auc roc曲线下面积
    accuracy = metrics.accuracy_score(tag_test, y_test_pred)  # accuracy准确率
    Rec = metrics.recall_score(tag_test, y_test_pred)  # recall 召回率
    Prec = metrics.precision_score(tag_test, y_test_pred)  # precision 精度
    f1 = metrics.f1_score(tag_test, y_test_pred)  # f1 score f1分数
    print("1-Prec: %.2f%%" % (Prec * 100.0))
    print("1-Recall: %.2f%%" % (Rec * 100.0))
    print("F1: %.2f%%" % (f1 * 100.0))
    print("auc: %.2f%%" % (auc * 100.0))
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    precision_1 += Prec
    recall_1 += Rec


# 五则取平均
mean_train_accuracy = total_train_accuracy / 5
mean_test_accuracy = total_test_accuracy / 5
print("//////////////////////////////////////////////")
print("Average train accuracy: ", mean_train_accuracy)
print("Average test accuracy: ", mean_test_accuracy)
print("//////////////////////////////////////////////")
mean_precision_0 = precision_0 / 5
mean_recall_0 = recall_0 / 5
print("Average precision_0: ", mean_precision_0)
print("Average recall_0: ", mean_recall_0)
mean_precision_1 = precision_1 / 5
mean_recall_1 = recall_1 / 5
print("Average precision_1: ", mean_precision_1)
print("Average recall_1: ", mean_recall_1)
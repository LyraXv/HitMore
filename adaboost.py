import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# 加载三个文件排名列表
df1 = pd.read_csv('file1.csv')
df2 = pd.read_csv('file2.csv')
df3 = pd.read_csv('file3.csv')

# 通过filename合并数据
merged_df = pd.merge(pd.merge(df1, df2, on='File name', suffixes=('_1', '_2')), df3, on='File name')
merged_df.rename(columns={'File rank': 'File rank_3', 'Suspicious score': 'Suspicious score_3'}, inplace=True)

# 特征和目标选择
X = merged_df[['File rank_1', 'File rank_2', 'File rank_3', 'Suspicious score_1', 'Suspicious score_2', 'Suspicious score_3']]
# We'll use a weighted average of the ranks as a simple target for demonstration
merged_df['average_rank'] = merged_df[['File rank_1', 'File rank_2', 'File rank_3']].mean(axis=1)
y = merged_df['average_rank']

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 初始化adaboost
base_estimator = DecisionTreeRegressor(max_depth=4)
ada_regressor = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=50, random_state=42)

# 训练模型
ada_regressor.fit(X_scaled, y)

# 预测最终排名
merged_df['final_rank'] = ada_regressor.predict(X_scaled)

# 排序
final_ranking = merged_df[['File name', 'final_rank']].sort_values(by='final_rank')

# 输出
final_ranking.to_csv('final_ranking.csv', index=False)

#print("Final ranking has been saved to 'final_ranking.csv'.")

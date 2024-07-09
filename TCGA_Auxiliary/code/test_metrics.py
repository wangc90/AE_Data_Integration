from sklearn.metrics import silhouette_score ,calinski_harabasz_score, davies_bouldin_score
import numpy as np
import pandas as pd

feature = np.loadtxt('../../data/Survival_data/5BRCA/final_embedding.txt')
# feature = pd.read_csv('../../data/Survival_data/5BRCA/lowDim=20_alpha=1_gamma=10_X.csv', header=None)
# feature = pd.read_csv('../../data/Survival_data/5BRCA/mdicc_embedding.csv', header=None)
# y_pred = np.loadtxt('../../data/Survival_data/18LUAD/mdicc_cluster_label_3.txt')
y_pred = pd.read_csv('../../data/Survival_data/5BRCA/cluster_label_5.csv', header=None)
feature = np.array(feature)
y_pred = np.array(y_pred)
print(feature.shape)
print(y_pred.shape)

score_feature1 = silhouette_score(feature, y_pred, metric='cosine')  # 计算轮廓系数
score_feature2 = calinski_harabasz_score(feature, y_pred)  # 计算CH分数
score_feature3 = davies_bouldin_score(feature, y_pred)  # 计算戴维森堡丁指数(DBI)

print(score_feature1)
print(score_feature2)
print(score_feature3)
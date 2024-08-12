from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

"""
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。常见选项有 'single'、'complete'、'average'、'weighted'、'centroid' 和 'ward'
2. method是指计算类间距离的方法。常见选项有 'euclidean'、'cityblock'、'cosine'、'pearson' 和 'jaccard'。

linkage() 函数返回一个链接矩阵,这是一个形状为 (n-1, 4) 的 2D numpy 数组,其中 n 是数据点的数量。链接矩阵的每一行都表示两个聚类的合并,包含以下信息:
被合并的第一个聚类的索引(范围为 [0, n-1]).
被合并的第二个聚类的索引(范围为 [0, n-1]).
被合并的两个聚类之间的距离。
合并后聚类中的数据点数量。
"""
"""
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
"""

X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
Z=linkage(X,method='ward')
f=fcluster(Z,4,"distance")
print(f)
plt.figure(figsize=(5,3))   #建立5*3的图纸
dendrogram(Z)   #可视化层次聚类树状图
print(Z)
plt.show()
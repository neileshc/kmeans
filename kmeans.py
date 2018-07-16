%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist
import xlrd
import pandas as pd
import numpy as np

xl = pd.ExcelFile("myXLData.xlsx")
df = xl.parse("Sheet1")
print list(df)

# Getting the values and scatter plotting it
f1 = df_filter['col1'].values
f2 = df_filter['col2'].values
x = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=10)
plt.xlabel('col1')
plt.ylabel('col2')

# Elbow analysis
from sklearn.cluster import KMeans
sum_squared_errors = {}
for k in range(1, 30):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_filter)
    sum_squared_errors[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sum_squared_errors.keys()), list(sum_squared_errors.values()))
plt.xlabel("Number of cluster")
plt.ylabel("sum_squared_errors")
plt.show()

# Number of clusters
kmeans = KMeans(n_clusters=5)
# Fitting the input data
kmeans = kmeans.fit(df_filter)
# Getting the cluster labels
labels = kmeans.predict(df_filter)
# Centroid values
centroids = kmeans.cluster_centers_
# vfunc = np.vectorize(around(2))
# vfunc(centroids)
np.set_printoptions(suppress=True)
print centroids
# print np.round(centroids, 4)

# overlay centroids on cluster scatterplot
# print df.iloc[0, :]
plt.scatter(df_filter.iloc[:, 0], df_filter.iloc[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, alpha=0.8);
plt.xlabel("col1")
plt.ylabel("col2")
# plt.ylim(0, 2000)
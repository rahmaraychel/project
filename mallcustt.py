# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score,classification_report
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.cluster import KMeans



# #load dataset

# customer_data=pd.read_csv('Mall_customers.csv')
# print(customer_data.head())

# x=customer_data.iloc[:,[3,4]].values
# print(x)
# wcss=[]
# for i in range(1,11):
#     Kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
#     Kmeans.fit(x)
#     wcss.append(Kmeans.inertia_)
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

customer_data=pd.read_csv('Mall_Customers.csv')
print(customer_data.head())

X= customer_data.iloc[:,[3,4]].values
print(X)


wcss = []
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    #elbow geaph
    sns.set()
plt.plot(range(1, 11), wcss, marker='o')
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

cluster=5
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
y=kmeans.fit_predict(X)
print(y)

cluster=0,1,2,3,4
plt.figure(figsize=(8,8))
plt.scatter(X[y==0,0],X[y==0,1],s=50,c='blue',label='cluster1')
plt.scatter(X[y==1,0],X[y==1,1],s=50,c='green',label='cluster2')
plt.scatter(X[y==2,0],X[y==2,1],s=50,c='pink',label='cluster3')
plt.scatter(X[y==3,0],X[y==3,1],s=50,c='black',label='cluster4')
plt.scatter(X[y==4,0],X[y==4,1],s=50,c='grey',label='cluster5')
    
 #centroids   
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='red',label='centroids')
plt.title
plt.xlabel
plt.show()

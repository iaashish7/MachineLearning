#! /usr/bin/python3
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing

def davies_bouldin(X, labels, cluster_ctr):
    clusters = list(set(labels))
    num_clusters = len(clusters)
    num_items_in_clusters = [0]*num_clusters
    for i in range(len(labels)):
        num_items_in_clusters[labels[i]] += 1
    max_num = -9999
    for i in range(num_clusters):
        s_i = inside_cluster_dist(X, labels, clusters[i], num_items_in_clusters[i], cluster_ctr[i])
        for j in range(num_clusters):
            if(i != j):
                s_j = inside_cluster_dist(X, labels, clusters[j], num_items_in_clusters[j], cluster_ctr[j])
                m_ij = np.linalg.norm(cluster_ctr[clusters[i]]-cluster_ctr[clusters[j]])
                r_ij = (s_i + s_j)/m_ij
                if(r_ij > max_num):
                    max_num = r_ij
    return max_num

def inside_cluster_dist(X, labels, cluster, num_items_in_cluster, centroid):
    total_dist = 0
    for k in range(num_items_in_cluster):
        dist = np.linalg.norm(X[labels==cluster]-centroid)
        total_dist = dist + total_dist
    return total_dist/num_items_in_cluster

def handle_non_numeric_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_values = {}
        def convert_to_int(val):
            return text_digit_values[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_values:
                    text_digit_values[unique] = x
                    x+=1
            df[column] = list(map(convert_to_int,df[column]))
    return df

df = pd.read_csv("Round2_Dataset_G2.csv")
df.drop(['Customer','Effective To Date'],1,inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0,inplace = True)
df = handle_non_numeric_data(df)
#print(df.head())
df.to_csv('Round2_Output_G2_AashishMisraa.csv')
clf = KMeans(n_clusters=10)
X = np.array(df.astype(float))
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_
DBI = davies_bouldin(X,labels,centroids)
print(DBI)   #This will return the Davies Bouldin Index which is coming out to be approx for this Round2_Dataset_G2
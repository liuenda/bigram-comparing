# coding: utf-8
import time
import pandas as pd
import numpy as np
import time
from sklearn.cluster import KMeans

start_time = time.time()
np.random.seed(42)

def read_vecs(lang_name):
	filename='./output/good_vecs_'+lang_name+'.csv'
	df=pd.read_csv(filename)
	return df

vecs_en=read_vecs('en')
vecs_jp=read_vecs('jp')

kmeans_en=KMeans(init='k-means++',n_clusters=10,n_init=10)
kmeans_en.fit(vecs_en.drop('en',axis=1))
kmeans_en_labels = kmeans_en.labels_
kmeans_en_centers = kmeans_en.cluster_centers_
print "English clustering results:", kmeans_en_labels

kmeans_jp=KMeans(init='k-means++',n_clusters=10,n_init=10)
kmeans_jp.fit(vecs_jp.drop('jp',axis=1))
kmeans_jp_labels = kmeans_jp.labels_
kmeans_jp_centers = kmeans_jp.cluster_centers_
print "Japanese clustering results:", kmeans_jp_labels

print("--- %s seconds ---" % (time.time() - start_time))
# coding: utf-8
import time
import pandas as pd
import numpy as np
import time
from sklearn.cluster import KMeans

start_time = time.time()
np.random.seed(42)
k=10
output_dir='output/cluster-scikit/'

def read_vecs(lang_name):
	filename='./output/good_vecs_'+lang_name+'.csv'
	print "---Reading the word2vec vectors in ",lang_name," from ",filename,"---"
	df=pd.read_csv(filename)
	return df

def save_cluster(vecs_en,kmeans_en_labels,lang_name,output_dir):
	# Save the result as CSV
	#columns_name=lang_name+'10'
	columns_name='k10'
	en10_good=pd.DataFrame(kmeans_en_labels,columns=[columns_name])
	en10_good[lang_name]=vecs_en[lang_name]
	en10_good=en10_good[[lang_name,columns_name]]
	en10_good.to_csv(output_dir+lang_name+str(k)+'_good.csv',index=False)


vecs_en=read_vecs('en')
vecs_jp=read_vecs('jp')

# kmeans_en=KMeans(init='k-means++',n_clusters=k,n_init=10)
kmeans_en=KMeans(init='k-means++',n_clusters=k,n_init=1)
kmeans_en.fit(vecs_en.drop('en',axis=1))
# Rather than cluster number from 1 to k, sci-kit learn will choose 
# 0 to k-1 as the cluster name, so we want them increase 1
kmeans_en_labels = kmeans_en.labels_+1 
kmeans_en_centers = kmeans_en.cluster_centers_
print "English clustering results:", kmeans_en_labels
# save the result to CSV
save_cluster(vecs_en,kmeans_en_labels,'en',output_dir)

# kmeans_jp=KMeans(init='k-means++',n_clusters=k,n_init=10)
kmeans_jp=KMeans(init='k-means++',n_clusters=k,n_init=1)
kmeans_jp.fit(vecs_jp.drop('jp',axis=1))
# Rather than cluster number from 1 to k, sci-kit learn will choose 
# 0 to k-1 as the cluster name, so we want them increase 1
kmeans_jp_labels = kmeans_jp.labels_+1
kmeans_jp_centers = kmeans_jp.cluster_centers_
print "Japanese clustering results:", kmeans_jp_labels
# save the result to CSV
save_cluster(vecs_jp,kmeans_jp_labels,'jp',output_dir)


print("--- %s seconds ---" % (time.time() - start_time))
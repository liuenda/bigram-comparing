# coding: utf-8
import time
import pandas as pd
import numpy as np
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

def save_centroid(kmeans_centers,lang_name):

	centroid_filename = "output/cluster-scikit/centroid_" + lang_name + str(k) + ".csv"
	df_centers=pd.DataFrame(kmeans_centers)
	df_centers.index = range(1,k+1)
	print "DEBUG: kmeans_centers is:"
	print np.shape(kmeans_centers)
	print pd.DataFrame(kmeans_centers)

	df_centers.to_csv(centroid_filename)

	return df_centers


if __name__ =="__main__":
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
	
	# save centroid to "centroid_en[k].csv"
	save_centroid(kmeans_en_centers,'en')


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

	# save centroid to "centroid_en[k].csv"
	save_centroid(kmeans_jp_centers,'jp')

	print("--- %s seconds ---" % (time.time() - start_time))
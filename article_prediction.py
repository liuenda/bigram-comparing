# coding:utf-8
from __future__ import division
import pandas as pd
import numpy as np
import time
import sys
import experiment
import getVector
from gensim.models import *
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import ast
from ast import literal_eval as make_tuple
start_time = time.time()

# pd.set_option('display.height', 1000)
# pd.set_option('display.max_rows', 100)
# pd.set_option('display.max_columns', 200)
# pd.set_option('display.width', 200)
# pd.reset_option('all')

k=30
dim=200
nan=np.empty(dim)
n_rows=20
repeat=50

input_article_en='articles/en999.txt'
input_article_jp='articles/jp999.txt'
input_mapping='output/mapping/mapping_en_'+str(k)+'.csv'
dir_cluster_center='output/cluster-skmeans/'
model_name_en = "../modeling/model-en/W2Vmodle.bin"
model_name_jp = "../modeling/model-jp/W2Vmodle.bin"

log_filename1='articles/log1.txt'
output_unmatch=open(log_filename1,'w')
# log_filename2='article/log2.txt'
# logout=open(log_filename2,'w')

info=""

model_en = Word2Vec.load(model_name_en)
model_jp = Word2Vec.load(model_name_jp)

def clean_en(article):
	tokens=article.split()
	stopwords=[]
	tokens_clean=['' if word in stopwords else word for word in tokens]
	return ' '.join(tokens_clean)

def clean_jp(article):
	tokens=article.split()
	stopwords=['する','なる']
	tokens_clean=['' if word in stopwords else word for word in tokens]
	return ' '.join(tokens_clean)

# Call mapping_word
def mapping_artile(article,model,wnl):
	tokens=article.split()
	tokens_mapping=[mapping_word(word,model,wnl) for word in tokens]
	# print "DEBUG: Finish 1 line-------------"
	return tokens_mapping

# Call mapping_word # much more slower, although there is no loop
def mapping_artile2(article,model,wnl):
	tokens=article.split()
	# tokens_mapping=[mapping_word(word,model,wnl) for word in tokens]
	s_tokens=pd.Series(tokens)
	tokens_mapping=s_tokens.apply(mapping_word,args=(model,wnl))
	# print "DEBUG: Finish 1 line-------------"
	return tokens_mapping

# Call: get_vector
# Find the nearest en-cluster for a given word
def mapping_word(word,model,wnl):
	# 1. get the center for each en-cluster
	df_center_en=experiment.find_cluster_center(dir_cluster_center,'en')
	
	# 2. find the word2vec expression
	vec=get_vector(word,model,wnl)
	# print "DEBUG: ",vec
	
	if np.all(vec!=nan):
		# 3. calculate the similarity matrix
		similarity_matrix_en = \
			np.array(df_center_en).dot(vec) # ????? Have a check!
		# print "DEBUG, similarity_matrix_en ="
		# print "with shape of ", np.shape(similarity_matrix_en)
		
		# 4. Get the maximum one that can present this cluster
		cluster_number=similarity_matrix_en.argmax()+1
		return cluster_number
	else:
		print "DEBUG: "
		return None

def get_vector(word,model,wnl):
	word=word.rstrip() # remove all '\n' and '\r'
	# word=word.lower()
	baseform=getVector.getBase(word,wnl)
	# print "DEBUG: ",model['good']
	# print "DEBUG: baseform= ", baseform
	try:
		vecW=model[baseform] #!!!Maybe the word is not existed
	except Exception,e:
		# info=''
		# counter_NaN+=1 #increase 1 to NaN counter
		# info+=repr(e)+"\n" #create log information
		# logout.write(info) #write log information to log file				
		#new 3.15: generate a useless list for deleting in the next stage
		output_unmatch.write(word) # no \n is needed since the 
		output_unmatch.write('\n')
		print "---Warning: Word ["+word+"] Vector Not Found ---"
		return nan
	else:
		vecW=getVector.vecNorm(vecW) #Normalized the raw vector
		# print "(the new length of the vector is:",LA.norm(vecW),")"
		# info+=baseform+": OK!\n" #create log information
		# logout.write(info) #write log information to log file
		# fout.write(rawVoc) #add in 16/3/17
		# good_list.append(rawVoc)
		#append the new vector to the matrix
		#if the vector is the first element in the matrix: 'good_vecs', reshape it
		return vecW

def conter_to_vector(group_list):
	s=pd.Series(group_list)
	s_counter=s.value_counts()
	dic=dict(zip(range(1,k+1),[0]*k))
	dic.update(s_counter.to_dict())
	return dic

def map_to_jp_vector(vector_en,df_mapping):
	vector_jp=[0]*k
	# For any one freqency list (such as vector_en=[1,54,3,13,...3])
	# For each one element in the frequence list, such as frequency=1:
	for index,frequency in enumerate(vector_en):
		# print "DEBUG: index=",index
		# print "DEBUG: frequency=",frequency
		# Get the mapping information for cluster 1, mapping=(15,7,3,1)
		mapping=df_mapping.iloc[index].mapping_parsed
		# print "DEBUG: mapping=",mapping
		similarity=df_mapping.iloc[index].max_similarity
		# print "DEBUG: similarity=",similarity
		# For each element in the mapping, such as the cluster_name=15
		# similarity=1
		for cluster_name in mapping:
			# print "DEBUG: cluster_name=",cluster_name
			vector_jp[cluster_name-1]+=similarity*frequency
			# print "DEBUG: vector_jp=",vector_jp
	return vector_jp

def repeat_test(df_accuracy,df_accuracy_top5,repeat,df_mapping):
	for index_test,test in enumerate(range(0,repeat)):
		df_en_clean=pd.read_csv("articles/en999_mapped_"+str(k)+".csv").sample(n_rows).reset_index()
		df_en_clean['transformation_en']=df_en_clean['transformation_en'].apply(lambda x:ast.literal_eval(x))
		random_number=df_en_clean['index']
		print "The random English article number: "
		print index_test,random_number.values

		# df_jp_clean['transformation_jp']=\
			# df_jp_clean.jp_article.apply(mapping_artile,args=(model_jp,wnl))
		# df_jp_clean.to_csv("articles/jp999_mapped_"+str(k)+".csv",index=False)
		df_jp_clean=pd.read_csv("articles/jp999_mapped_"+str(k)+".csv").iloc[random_number].reset_index()
		df_jp_clean['transformation_jp']=df_jp_clean['transformation_jp'].apply(lambda x:ast.literal_eval(x))

		# Call conter_to_vector()
		df_en_clean['f_vector']=df_en_clean['transformation_en'].apply(conter_to_vector)
		df_en_clean['f_vector']=df_en_clean['f_vector'].apply(lambda x:x.values())

		df_jp_clean['f_vector']=df_jp_clean['transformation_jp'].apply(conter_to_vector)
		df_jp_clean['f_vector']=df_jp_clean['f_vector'].apply(lambda x:x.values())

		# Call map_to_jp_vector()
		df_en_clean['en2jp_projection']=\
			df_en_clean['f_vector'].apply(map_to_jp_vector,args=(df_mapping,))

		#------------------
		# EN: Normalize the en_to_jp_projection
		df_en_clean['en2jp_projection_norm']=\
			df_en_clean['en2jp_projection'].apply(getVector.vecNorm)

		# JP: Normalize the f_vector
		df_jp_clean['f_vector_norm']=\
			df_jp_clean['f_vector'].apply(getVector.vecNorm)

		#------------------
		# Normalize the en_to_jp_projection
		df_en_vector_matrix=df_en_clean['en2jp_projection_norm'].apply(pd.Series)

		# Normalize the en_to_jp_projection
		df_jp_vector_matrix=df_jp_clean['f_vector_norm'].apply(pd.Series)

		# Calculate the most similar Japanese article
		similarity_matrix_jp=\
			np.array(df_en_vector_matrix).dot(np.array(df_jp_vector_matrix).T)

		# @top1
		prediction_jp=similarity_matrix_jp.argmax(axis=1) # --> maximum inddex for each row

		# @top5
		prediction_jp_top5=similarity_matrix_jp.argsort(axis=1)[:,-5:]

		df_result=pd.DataFrame(df_en_clean['en_article'])
		df_result['prediction_jp_name']=pd.Series(prediction_jp)
		df_result['prediction_jp_article']=df_jp_clean.iloc[prediction_jp].reset_index().jp_article
		
		# Sequencial version (head: n_rows)
		df_result['real_jp_name']=pd.Series(range(0,n_rows))

		df_result['real_jp_article']=df_jp_clean.jp_article
		df_result['evaluation']=(df_result.prediction_jp_name==df_result.real_jp_name)

		# @top5
		df_result['prediction_jp_top5_name']=pd.Series(prediction_jp_top5.tolist())
		df_result['evaluation_top5']=\
			df_result.real_jp_name.apply(lambda x,y:(x in y.iloc[x]),args=(df_result.prediction_jp_top5_name,))


		# print "The expectation is ",df_result.evaluation.value_counts()
		# @top1
		df_accuracy[index_test]=df_result.evaluation.value_counts()
		# @top5
		df_accuracy_top5[index_test]=df_result.evaluation_top5.value_counts()

	return df_accuracy,df_accuracy_top5



if __name__ == "__main__":

	wnl = WordNetLemmatizer()

	model_en = Word2Vec.load(model_name_en)
	model_jp = Word2Vec.load(model_name_jp)

	df_en=pd.read_table(input_article_en,names=["en_article"])
	df_jp=pd.read_table(input_article_jp,names=["jp_article"])

	df_en_clean=df_en.applymap(clean_en)
	df_jp_clean=df_jp.applymap(clean_jp)

	df_mapping=pd.read_csv(input_mapping)
	df_mapping['mapping_parsed']=df_mapping.mapping.map(lambda x: make_tuple(x))

	# df_en_clean['transformation_en']=\
	# 	df_en_clean.en_article.apply(mapping_artile,args=(model_en,wnl))
	# df_en_clean.to_csv("articles/en999_mapped_"+str(k)+".csv",index=False)
	df_en_clean=pd.read_csv("articles/en999_mapped_"+str(k)+".csv").sample(n_rows).reset_index()
	df_en_clean['transformation_en']=df_en_clean['transformation_en'].apply(lambda x:ast.literal_eval(x))
	random_number=df_en_clean['index']
	# print "The random English article number: "
	# print random_number

	# df_jp_clean['transformation_jp']=\
		# df_jp_clean.jp_article.apply(mapping_artile,args=(model_jp,wnl))
	# df_jp_clean.to_csv("articles/jp999_mapped_"+str(k)+".csv",index=False)
	df_jp_clean=pd.read_csv("articles/jp999_mapped_"+str(k)+".csv").iloc[random_number].reset_index()
	df_jp_clean['transformation_jp']=df_jp_clean['transformation_jp'].apply(lambda x:ast.literal_eval(x))

	# Call conter_to_vector()
	df_en_clean['f_vector']=df_en_clean['transformation_en'].apply(conter_to_vector)
	df_en_clean['f_vector']=df_en_clean['f_vector'].apply(lambda x:x.values())

	df_jp_clean['f_vector']=df_jp_clean['transformation_jp'].apply(conter_to_vector)
	df_jp_clean['f_vector']=df_jp_clean['f_vector'].apply(lambda x:x.values())

	# Call map_to_jp_vector()
	df_en_clean['en2jp_projection']=\
		df_en_clean['f_vector'].apply(map_to_jp_vector,args=(df_mapping,))

	#------------------
	# EN: Normalize the en_to_jp_projection
	df_en_clean['en2jp_projection_norm']=\
		df_en_clean['en2jp_projection'].apply(getVector.vecNorm)

	# JP: Normalize the f_vector
	df_jp_clean['f_vector_norm']=\
		df_jp_clean['f_vector'].apply(getVector.vecNorm)

	#------------------
	# Normalize the en_to_jp_projection
	df_en_vector_matrix=df_en_clean['en2jp_projection_norm'].apply(pd.Series)

	# Normalize the en_to_jp_projection
	df_jp_vector_matrix=df_jp_clean['f_vector_norm'].apply(pd.Series)

	# Calculate the most similar Japanese article
	similarity_matrix_jp=\
		np.array(df_en_vector_matrix).dot(np.array(df_jp_vector_matrix).T)

	# @top1
	prediction_jp=similarity_matrix_jp.argmax(axis=1) # --> maximum inddex for each row

	# @top5
	prediction_jp_top5=similarity_matrix_jp.argsort(axis=1)[:,-5:]

	df_result=pd.DataFrame(df_en_clean['en_article'])
	df_result['prediction_jp_name']=pd.Series(prediction_jp)
	df_result['prediction_jp_article']=df_jp_clean.iloc[prediction_jp].reset_index().jp_article


	# Sequencial version (head: n_rows)
	df_result['real_jp_name']=pd.Series(range(0,n_rows))

	df_result['real_jp_article']=df_jp_clean.jp_article
	df_result['evaluation']=(df_result.prediction_jp_name==df_result.real_jp_name)

	# @top5
	df_result['prediction_jp_top5_name']=pd.Series(prediction_jp_top5.tolist())
	df_result['evaluation_top5']=\
		df_result.real_jp_name.apply(lambda x,y:(x in y.iloc[x]),args=(df_result.prediction_jp_top5_name,))

	print "The expectation is ",df_result.evaluation.value_counts() 
	output_unmatch.close()	

	# @ top1 Accuracy calculation
	df_accuracy = pd.DataFrame(index=[False,True])
	df_accuracy['base']=pd.DataFrame(df_result.evaluation.value_counts())
	
	# @top5 Accuracy calculation
	df_accuracy_top5 = pd.DataFrame(index=[False,True])
	df_accuracy_top5['base']=pd.DataFrame(df_result.evaluation_top5.value_counts())

	# Repeat the test 
	df_accuracy_final, df_accuracy_top5_final=repeat_test(df_accuracy,df_accuracy_top5,repeat,df_mapping)

	# print out the results
	print df_accuracy_final
	print df_accuracy.sum(axis=1)
	print "the expectation is: "
	# print df_accuracy.sum(axis=1).iloc[1]/df_accuracy.sum(axis=1).iloc[0]*100,"%"
	print df_accuracy.sum(axis=1).iloc[1]/(repeat)	
	print "maximum prediction level: ",df_accuracy.iloc[1].max()

		# print out the results
	print df_accuracy_top5_final
	print df_accuracy_top5.sum(axis=1)
	print "the expectation is: "
	# print df_accuracy.sum(axis=1).iloc[1]/df_accuracy.sum(axis=1).iloc[0]*100,"%"
	print df_accuracy_top5.sum(axis=1).iloc[1]/(repeat)	
	print "maximum prediction level: ",df_accuracy_top5.iloc[1].max()

	print("--- %s seconds ---" % 
		(time.time() - start_time))
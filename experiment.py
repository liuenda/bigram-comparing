# coding: utf-8
import time
import pandas as pd
import numpy as np
import baseline
from ast import literal_eval as make_tuple

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

start_time = time.time()

k = 10
# output_dir='output/cluster-scikit/'
output_dir='output/cluster-skmeans/'

def load_data():
	test_en_filename = "output_newwords/"+"good_vecs_en.csv"
	test_jp_filename = "output_newwords/"+"good_vecs_jp.csv"

	training_en_filename = "output/"+"good_vecs_en.csv"
	training_jp_filename = "output/"+"good_vecs_jp.csv"

	old_dict_filename = "output/" + "good_dic.csv"
	new_dict_filename = "output_newwords/"+"good_dic.csv"

	df_test_en = pd.read_csv(test_en_filename)
	df_test_jp = pd.read_csv(test_jp_filename)

	df_training_en = pd.read_csv(training_en_filename)
	df_training_jp = pd.read_csv(training_jp_filename)

	df_old_dict = pd.read_csv(old_dict_filename)
	df_new_dict = pd.read_csv(new_dict_filename)

	return df_test_en, df_test_jp, df_training_en, df_training_jp, df_old_dict, df_new_dict


# This will return a similarity matrix with shape of df_test(x,)*df_training(,y) --> x*y
def get_similarity(df_training,df_test,lang_name):
	similarity_matrix=np.array(df_training.drop(lang_name,axis=1)).dot(np.array(df_test.drop(lang_name, axis=1).T))
	return similarity_matrix


def get_top1_similar_words(similarity_matrix,axis_=0):
	s_similarity_top_1 = pd.DataFrame(similarity_matrix).idxmax(axis=axis_)
	# s_similarity_top_1 = pd.DataFrame(similarity_matrix).idxmin(axis=axis_)
	# print "DEBUG: s_similarity_top_1 is:"
	# print s_similarity_top_1
	return s_similarity_top_1


def get_translation(s_similarity_top_1,df_dict,df_test,lang_name):
	top_translation=df_dict.ix[s_similarity_top_1].reset_index()
	# print "DEBUG: top_translation is:"
	# print top_translation

	top_translation['word_to_predict']=df_test[lang_name]
	# print "DEBUG: top_translation (new) is:"
	# print top_translation

	return top_translation


def baseline_method(df_test_en, df_test_jp, df_training_en, df_training_jp, df_old_dict, df_new_dict):
	
	# From "unknown" new en word to jp word
	# This will return a similarity matrix with shape of df_test(x,)*df_training(,y)
	# (94*200)*(46*200).T --> (94*46)
	similarity_matrix_en = get_similarity(df_training_en,df_test_en,'en')
	print "DEBUG: the shape of the similar_word_en is (should be 94*46):"
	print np.shape(similarity_matrix_en)
	
	# This will return a top-1 candidates length of 46
	# Here the content of s_similarity_top_1 is from 0 to 95 (all 96)
	s_similarity_top_1 = get_top1_similar_words(similarity_matrix_en)
	print "DEBUG: the shape of the s_similarity_top_1 is (should be ,46):"
	print np.shape(s_similarity_top_1)

	# This will 1st, convert english to Japanse words, with index reset
	# index, jp, en, word_to_predict
	top_translation = get_translation(s_similarity_top_1,df_old_dict,df_test_en,'en')
	# print "DEBUG: top_translation is"
	# print top_translation

	# This will return a similarity matrix with shape of df_training_jp_select(x,)*df_test_jp(,y)
	# (46*200)*(46*200).T --> (46*46)
	df_training_jp_select = df_training_jp.ix[top_translation['index']]
	similarity_matrix_jp = get_similarity(df_training_jp_select,df_test_jp,'jp')
	print "DEBUG: the shape of the similar_word_en is (should be 46*46):"
	print np.shape(similarity_matrix_jp)

	s_similarity_top_2=get_top1_similar_words(similarity_matrix_jp,axis_=1)
	# print "DEBUG: the s_similarity_top_2 is:"
	# print s_similarity_top_2

	print "DEBUG: df_new_dict.ix[s_similarity_top_2] is:"
	print df_new_dict.ix[s_similarity_top_2]
	top_translation['jp_predicting_results']=df_new_dict.ix[s_similarity_top_2].reset_index().jp
	# print df_new_dict.ix[s_similarity_top_2]['jp']
	# print top_translation

	top_translation[["reference*","real_translation"]]=df_new_dict
	print "the final result is:"
	print top_translation
	# Calculate the accuracy
	print top_translation['jp_predicting_results'] == top_translation['real_translation']


# call: find_cluster_center()
# call: find_group_center()
def proposal_method(df_test_en, df_test_jp, df_training_en, df_training_jp, df_old_dict, df_new_dict):
	# Load mapping (en cluster VS jp groups)
	mapping_filename = "output/mapping/mapping_en_"+str(k)+".csv"
	df_mapping = pd.read_csv(mapping_filename)

	# Find center for en-cluster  (retreive)
	cluster_centroid_dir=  output_dir
	df_center_en = find_cluster_center(cluster_centroid_dir,'en')

	# Find center for jp-cluster  (retreive)
	cluster_centroid_dir=  output_dir
	df_center_jp = find_cluster_center(cluster_centroid_dir,'jp')

	# Find the center of the jp-groups through df_mapping (mapping)
	df_mapping = find_group_center(df_mapping,df_center_jp)

	# Finish preparation -----------------------------

	# Find the closest en-cluster for the new-words (1)
	# Get similarity matrix
	similarity_matrix_en = np.array(df_center_en).dot(
		np.array(df_test_en.drop('en', axis=1).T))
	print "DEBUG, similarity_matrix_en ="
	# print similarity_matrix_en
	print "with shape of ", np.shape(similarity_matrix_en)

	# Find the closest en-cluster for the new-words (2)
	# Find the top-1 en-cluster
	# Here, deferent from baseline method, the increasement by 1 is needed??
	s_similarity_top_en = get_top1_similar_words(similarity_matrix_en)
	print "DEBUG, s_similarity_top_en = "
	print s_similarity_top_en


	# Attach the closest en-cluster to the new-wordlist
	df_result = pd.DataFrame(df_test_en.en)
	# Attach detailed information for df_result
	df_result['closest_en_cluster'] = s_similarity_top_en
	# Attach the mapped jp-groups
	df_result['mapping_parsed'] = df_mapping.ix[df_result.closest_en_cluster].reset_index(
		).drop(range(1,201),axis=1)['mapping_parsed']
	df_result[range(1,201)] = df_mapping.ix[df_result.closest_en_cluster].reset_index(
		)[range(1,201)]

	# find the closest jp group for each new jp word: (1)
	similarity_matrix_jp=np.array(df_result[range(1,201)]).dot(np.array(df_test_jp.drop('jp', axis=1).T))
	print "DEBUG, the shape of similarity_matrix_jp is"
	print np.shape(similarity_matrix_jp)

	#find the closest jp group for each new jp word: (2)
	s_similarity_top_jp = get_top1_similar_words(similarity_matrix_jp)
	print "DEBUG, the shape of s_similarity_top_jp"
	print s_similarity_top_jp
	print np.shape(s_similarity_top_jp)

	df_result['s_similarity_top_jp'] = s_similarity_top_jp
	df_result['jp_predicted'] = df_test_jp.ix[s_similarity_top_jp].reset_index()['jp']
	df_result = df_result.drop(range(1,201),axis=1)

	# Compare the anwser
	df_result['real_translation'] = df_test_jp['jp']
	df_result['accuracy'] = (df_result['real_translation'] == df_result['jp_predicted'])
	print "the df_result is:"
	print df_result

# Find the center of each cluster
def find_cluster_center(cluster_centroid_dir,lang_name):
	cluster_center_filename=cluster_centroid_dir + "centroid_" + lang_name + str(k) + ".csv"
	df_cluster_center = pd.read_csv(cluster_center_filename,index_col=0)
	# print "DEBUG: df_cluster_center is [" +  cluster_center_filename + "]" 
	# print df_cluster_center
	return df_cluster_center


# Call: calculate_from_tuple
def find_group_center(df_mapping,df_center):
	df_mapping['mapping_parsed']=df_mapping.mapping.map(lambda x: make_tuple(x))
	df_mapping[range(1,201)]=df_mapping.mapping_parsed.apply(
		calculate_from_tuple, args=(df_center,))
	print "DEBUG: df_mapping is: [containing the results of new centroid for groups]"
	print df_mapping
	return df_mapping

# *ONLY* CAN be called by find_group_center()
def calculate_from_tuple(t,df_center):
	group_center=0
	for tt in t:
		print "find the cluster center: ", tt 
		group_center += df_center.ix[tt]
	print "calculation for one tuple finished -----"
	print "the shape of this group center is: ", np.shape(group_center/len(t))
	print "the type of the group center is", type(group_center/len(t))
	print ""
	return group_center/len(t)


if __name__ == "__main__":

	df_test_en, df_test_jp, df_training_en, df_training_jp, df_old_dict, df_new_dict = load_data()
	
	baseline_method(df_test_en, df_test_jp, df_training_en, df_training_jp, df_old_dict, df_new_dict)
	
	proposal_method(df_test_en, df_test_jp, df_training_en, df_training_jp, df_old_dict, df_new_dict)

	print("--- %s seconds ---" % (time.time() - start_time))
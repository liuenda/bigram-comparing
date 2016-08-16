# coding: utf-8
import time
import pandas as pd
import numpy as np
import baseline

start_time = time.time()

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
	print "DEBUG: the s_similarity_top_2 is:"
	print s_similarity_top_2

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


	# baseline_result = get_translation(similarity_matrix_jp,df_new_dict,'jp')
	
	# print "the baseline_result is:"
	# print baseline_result
	# return baseline_result


def proposal_method(df_test_en, df_test_jp, df_training_en, df_training_jp, df_old_dict, df_new_dict):
	load_data()
	mapping_filename1 = "output/mapping/mapping_en_10.csv"
	df_mapping = pd.read_csv(mapping_filename1)


if __name__ == "__main__":
	df_test_en, df_test_jp, df_training_en, df_training_jp, df_old_dict, df_new_dict = load_data()
	
	baseline_method(df_test_en, df_test_jp, df_training_en, df_training_jp, df_old_dict, df_new_dict)
	# proposal_method(df_test_en, df_test_jp, df_training_en, df_training_jp, df_old_dict, df_new_dict)

	print("--- %s seconds ---" % (time.time() - start_time))
# coding: utf-8
import time
import pandas as pd
import getVector
import cross_check
import numpy as np


def split_file(input_dic_name,input_list_name_en,input_list_name_jp):
	df=pd.read_csv(input_dic_name) 
	df.en.to_csv(input_list_name_en,index=False,encoding='utf-8')
	df.jp.to_csv(input_list_name_jp,index=False,encoding='utf-8')

def fit_projection(E,J):
	JTJ=np.dot(J.T,J)
	# iJTJ=np.linalg.inv(np.matrix(JTJ))
	# print np.dot(JTJ,iJTJ) # This is now identical!!
	# print np.linalg.det(JTJ)  # The determinant of det(JTJ) is 0!
	# Which means there is no inverse matrix

	# Solutions: Using linear ridge regression 
	la=0.00001
	dim=200
	I= np.eye(dim)
	print "inv(JTJ)*JTJ=I?: "
	print (JTJ + la * I).dot(np.linalg.inv(JTJ + la * I))
	W= np.dot(np.linalg.inv(JTJ + la * I),J.T).dot(E)


# Input: a en-jp dictionary, the output directory, two word2vec model
# Output: --> words_en.txt, words_jp.text 
#		  --> log files, vecs_en.csv, vecs_jp.csv
# 		  --> good_vecs_en.csv, good_vecs_good_en.csv, good_dic.csv

# Here, for input dictionary files, only a CSV without columns names is expected!
# In addtion, the format of content is limited, only in ['jp','en'] format

def run(mode):

	if mode == 'baseline':
		print "start to run under the 'baseline' model [baseline.py]"
		output_dir='output_newwords/'
		input_dic_name="new_words/my_jp-en_short.csv"
	
	if mode == 'proposed':
		print "start to run under the 'proposed' model [baseline.py]"
		output_dir='output/'
		input_dic_name="input/dic_103.csv"

	input_modelname_en="../modeling/model-en/W2Vmodle.bin"
	input_modelname_jp="../modeling/model-jp/W2Vmodle.bin"



	# split dictionary--------------------------------------------
	# split the en-jp dictionary into two individual file in JP and in EN respectively
	input_list_name_en = output_dir+"words_en.txt" # Output of split_file()
	input_list_name_jp = output_dir+"words_jp.txt" # Output of split_file()
	split_file(input_dic_name, input_list_name_en, input_list_name_jp)



	# Fetch English retreive word2vec vectors----------------------
	output_filename = output_dir+"vecs_en.csv"
	log_filename1 = output_dir+"log/bad_list_en.txt"
	log_filename2 = output_dir+"log/getVector_log_en.txt"
	
	getVector.retrieve_vec(model_name=input_modelname_en, input_filename=input_list_name_en, 
		output_filename=output_filename, log_filename1=log_filename1, log_filename2=log_filename2)



	# Fetch Japanese retreive word2vec vectors-----------------------
	output_filename = output_dir+"vecs_jp.csv"
	log_filename1 = output_dir+"log/bad_list_jp.txt"
	log_filename2 = output_dir+"log/getVector_log_jp.txt"

	getVector.retrieve_vec_jp(model_name=input_modelname_jp, input_filename=input_list_name_jp, 
		output_filename=output_filename, log_filename1=log_filename1, log_filename2=log_filename2)


	# cross-check the JP, EN file--------------------------------
	input_filename3=output_dir+'vecs_jp.csv' # output_filename (jp)
	input_filename4=output_dir+'vecs_en.csv' # output_filename (en)

	output_filename1=output_dir+'good_vecs_jp.csv'
	output_filename2=output_dir+'good_vecs_en.csv'
	output_filename3=output_dir+'good_dic.csv'

	df_jp,df_en,df_good_dic = cross_check.cross_check(input_filename3,input_filename4,input_dic_name,
	output_filename1,output_filename2, output_filename3)

	E = df_en[range(1,201)]
	J = df_jp[range(1,201)]

	return df_jp,df_en,df_good_dic

if __name__ == '__main__':

	start_time=time.time()

	# start to run baseline method
	df_jp,df_en,df_good_dic = run(mode='baseline')

	# start to run (preprare) the proposed method
	df_jp,df_en,df_good_dic = run(mode='proposed')

	print("--- %s seconds ---" % (time.time() - start_time))

# coding: utf-8
import time
import pandas as pd
import getVector
import cross_check
import numpy as np


def split_file(input_filename,input_newword_en,input_newword_jp):
	df=pd.read_table(input_filename,names=["en","jp"])
	df.en.to_csv(input_newword_name_en,index=False,encoding='utf-8')
	df.jp.to_csv(input_newword_name_jp,index=False,encoding='utf-8')

def fix_projection(E,J):
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


if __name__ == '__main__':

	start_time=time.time()

	input_filename="new_words/my_jp-en_short.txt"
	input_modelname_en="../modeling/model-en/W2Vmodle.bin"
	input_modelname_jp="../modeling/model-jp/W2Vmodle.bin"
	input_newword_name_en = "output_newwords/new_words_en.txt" # Output of split_file()
	input_newword_name_jp = "output_newwords/new_words_jp.txt" # Output of split_file()

	# model_en = Word2Vec.load(input_filename2)
	# model_jp = Word2Vec.load(input_filename3)

	split_file(input_filename, input_newword_name_en, input_newword_name_jp)

	# For Japanese retreive word2vec vectors
	output_filename = "output_newwords/vecs_en.csv"
	log_filename1 = "output_newwords/log/bad_list_en.txt"
	log_filename2 = "output_newwords/log/getVector_log_en.txt"
	
	getVector.retrieve_vec(model_name=input_modelname_en, input_filename=input_newword_name_en, 
		output_filename=output_filename, log_filename1=log_filename1, log_filename2=log_filename2)


	# For Japanese retreive word2vec vectors
	output_filename = "output_newwords/vecs_jp.csv"
	log_filename1 = "output_newwords/log/bad_list_jp.txt"
	log_filename2 = "output_newwords/log/getVector_log_jp.txt"

	getVector.retrieve_vec_jp(model_name=input_modelname_jp, input_filename=input_newword_name_jp, 
		output_filename=output_filename, log_filename1=log_filename1, log_filename2=log_filename2)

	# cross-check the JP, EN file
	input_filename3='output_newwords/vecs_jp.csv'
	input_filename4='output_newwords/vecs_en.csv'

	# Here, only CSV file is expected!
	# In addtion, the format of content is limited, only in ['jp','en'] format
	input_filename5='new_words/my_jp-en_short.csv'

	output_filename1='output_newwords/good_vecs_jp.csv'
	output_filename2='output_newwords/good_vecs_en.csv'
	output_filename3='output_newwords/good_dic.csv'

	df_jp,df_en,df_good_dic = cross_check.cross_check(input_filename3,input_filename4,input_filename5,
	output_filename1,output_filename2, output_filename3,col_names=['en','jp'])

	E = df_en[range(1,201)]
	J = df_jp[range(1,201)]

	# similarity = np.dot(np.matrix(E),np.matrix(J).T)





	print("--- %s seconds ---" % (time.time() - start_time))

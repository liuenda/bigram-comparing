# coding: utf-8
import time
import pandas as pd
import getVector


def split_file(input_filename,input_newword_en,input_newword_jp):
	df=pd.read_table(input_filename,names=["en","jp"])
	df.en.to_csv(input_newword_name_en,index=False,encoding='utf-8')
	df.jp.to_csv(input_newword_name_jp,index=False,encoding='utf-8')


if __name__ == '__main__':

	start_time=time.time()

	input_filename="new_words/my_jp-en_short.txt"
	input_modelname_en="../modeling/model-en/W2Vmodle.bin"
	input_modelname_jp="../modeling/model-jp/W2Vmodle.bin"
	input_newword_name_en = "new_words_en.txt"
	input_newword_name_jp = "new_words_jp.txt"

	# model_en = Word2Vec.load(input_filename2)
	# model_jp = Word2Vec.load(input_filename3)

	output_filename = "output_newwords/vecs_en.csv"
	log_filename1 = "output_newwords/log/bad_list_en.txt"
	log_filename2 = "output_newwords/log/getVector_log_en.txt"

	split_file(input_filename, input_newword_name_en, input_newword_name_jp)
	
	getVector.retrieve_vec(model_name=input_modelname_en, input_filename=input_newword_name_en, 
		output_filename=output_filename, log_filename1=log_filename1, log_filename2=log_filename2)

	output_filename = "output_newwords/vecs_jp.csv"
	log_filename1 = "output_newwords/log/bad_list_jp.txt"
	log_filename2 = "output_newwords/log/getVector_log_jp.txt"

	getVector.retrieve_vec_jp(model_name=input_modelname_jp, input_filename=input_newword_name_jp, 
		output_filename=output_filename, log_filename1=log_filename1, log_filename2=log_filename2)


	print("--- %s seconds ---" % (time.time() - start_time))

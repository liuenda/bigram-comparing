# coding: utf-8 
# 这个脚本的作用是提取一个单词（日语或者英语）的word2vec表达式
# 这个脚本的目标是替代find_vecs_xx.py并且让自身模块化方便调用

import time
import sys
start_time=time.time()
import gensim, logging
import numpy as np
from gensim.models import *
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from numpy import linalg as LA
import pandas as pd
import csv

reload(sys)  
sys.setdefaultencoding('utf8')
wnl = WordNetLemmatizer()

def getBase(str,lm):
	str=lm.lemmatize(str, 'v')
	str=lm.lemmatize(str, 'n')
	return str;

def vecNorm(vector):
	vectorNorm=vector/LA.norm(vector)
	return vectorNorm

# An example of using this function:
# 	input_filename = "new_words/my_jp-en_short.txt"
# 	model_name_en = "../modeling/model-en/W2Vmodle.bin"
# 	model_name_jp = "../modeling/model-jp/W2Vmodle.bin"
# 	output_filename = "output_newwords/vecs_en.csv"
# 	log_filename1 = "output_newwords/Log/bad_list_en.txt"
# 	log_filename2="output_newwords/Log/getVector_log.txt"
def retrieve_vec(model_name, input_filename, output_filename, 
	log_filename1, log_filename2):
	print "starting to retrieve word2vec for english word list:"
	print "model_name (the name of the w2v model) = ",model_name
	print "input_filename (the word list) = ", input_filename
	print "output_filename (the word+vec results in CSV) = ",output_filename
	print "log_filename1 (the failed list) = ",log_filename1
	print "log_filename2 (the log) = ",log_filename2

	#Initialization
	logout=open(log_filename2,'w')
	output_unmatch=open(log_filename1,'w')
	model = Word2Vec.load(model_name)
	good_list=[]
	counter_NaN=0
	count_phrase=0
	count_word=0
	wnl = WordNetLemmatizer()
	dim=200

	# Start retreiving vectors
	with open(input_filename) as vocab_file:
		for index,vocab in enumerate(vocab_file):
			# debug
			print index

			rawVoc=vocab.rstrip() # remove all '\n' and '\r'
			vocab=rawVoc.lower()			
			info=""

			# If the line is a SINGLE word
			if " " not in vocab:
				count_word+=1
				print "word",vocab
				baseform=getBase(vocab,wnl)
				try:
					vecW=model[baseform] #!!!Maybe the word is not existed
				except Exception,e:
					counter_NaN+=1 #increase 1 to NaN counter
					info+=repr(e)+"\n" #create log information
					logout.write(info) #write log information to log file				
					#new 3.15: generate a useless list for deleting in the next stage
					output_unmatch.write(rawVoc) # no \n is needed since the 
					output_unmatch.write('\n')
				else:
					vecW=vecNorm(vecW) #Normalized the raw vector
					print "the new length of the vector is:"
					print LA.norm(vecW)
					info+=baseform+": OK!\n" #create log information
					logout.write(info) #write log information to log file
					# fout.write(rawVoc) #add in 16/3/17
					good_list.append(rawVoc)
					#append the new vector to the matrix
					#if the vector is the first element in the matrix: 'good_vecs', reshape it
					if index==0: 
						good_vecs=vecW.reshape(1,dim)
					else:
						good_vecs=np.concatenate((good_vecs,vecW.reshape(1,dim)),axis=0)

			# If the line is a PHRASE --------
			else:
				count_phrase+=1
				tokens = nltk.word_tokenize(vocab)
				tagged = nltk.pos_tag(tokens)
				vecP=np.zeros(dim)
				#print tagged
				modified=[]
				for vocabP in tagged:
					flag_nan=False
					if vocabP[1]!="IN": # ignore the IN components
						modified.append(vocabP[0])
						baseformP=getBase(vocabP[0],wnl)
						try:
							vec1word=model[baseformP] #!!!Maybe the word is not existed
						except Exception,e:
							#print repr(e)
							info+=repr(e)+" " #create log information
							counter_NaN+=1 #increase the nan counter
							flag_nan=True
							break #if exception occrus, stop the for loop

						else:
							vecP+=vec1word #add vectors
							info+=baseformP+": OK!; " 
							#print vec

				logout.write(info+"\n") #write log information to log file
				print "phrase:",modified #print the good phrase

				# if the phrase is all good, nomalize it and then save it to the goodlist file (fout) and save the array to good_vecs
				if flag_nan==False:
					#Normalize the added raw phrase vector to unit length
					vecP=vecNorm(vecP)
					print "the new length of the vector is:"
					print LA.norm(vecP)

					#fout.write(" ".join(modified)+"\n") #here is a problem which will produce one more line here
					# fout.write(rawVoc)
					good_list.append(rawVoc)
					if index==0:
						good_vecs=vecP.reshape(1,dim)
					else:
						good_vecs=np.concatenate((good_vecs,vecP.reshape(1,dim)),axis=0)
				else:
					#new 3.15: generate a useless list for deleting in the next stage
					output_unmatch.write(rawVoc)

		print "NaN words:",counter_NaN
		print "Pharses Number:",count_phrase
		print "all lines:",index+1
		print "all word:",count_word

		# Create a pandas Series and save
		df = pd.DataFrame(good_vecs,index=good_list)
		df.to_csv(output_filename,index=True,quoting=csv.QUOTE_NONNUMERIC,encoding='utf-8')

		output_unmatch.close()
		logout.close()

def retrieve_vec_jp(model_name, input_filename, output_filename, 
	log_filename1, log_filename2):
	
	# print "!!This function can only be run on the Unix system equiped with MeCab!!"
	# import MeCab

	# output_filename1=r'output_newwords/log/tag_mecab_jp.txt'
	# output_filename2=r'output_newwords/log/cleaned_tag_jp.txt'

	# output1=open(output_filename1,'w')
	# output2=open(output_filename2,'w')

	# tagger = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")

	# with open(input_filename) as data_file:
	# 	for (index,line) in enumerate(data_file):
	# 		#line=line.encode('utf-8','ignore')  # NO NEED!
	# 		node = tagger.parseToNode(line)
	# 		#index=0
	# 		line_tagged=[]
	# 		newLine=[]
	# 		while node:
	# 			word_tagged=(node.surface,node.feature)
	# 			line_tagged.append(word_tagged)
	# 			list_feature=node.feature.split(',')
	# 			if '動詞' in list_feature[0] or '名詞' in list_feature[0] or '接頭詞' in list_feature[0]:
	# 				if '数' not in list_feature[1] and '接尾' not in list_feature[1]:
	# 					if '*' not in list_feature[6]:
	# 						newLine.append(list_feature[6])
	# 			# if index==999:
	# 			# 	print list_feature[0]
	# 			node=node.next

	# 		output2.write(' '.join(newLine)+'\n')
	# 		# # debug
	# 		# print "the tagged new line is ",newLine

	# 		output1.write('\n'.join('_'.join(t) for t in line_tagged))
	# 		output1.write('\n\n\n')

	# # 以上代码有bug，如果重新启用，一定要注意！！！

	# output1.close()
	# output2.close()

	# 此阶段将人工手动删除 する 结尾的单词，之后可能会重新启用mecab

	retrieve_vec(model_name, input_filename, output_filename, 
	log_filename1, log_filename2)


if __name__ == '__main__':

	# See baseline.py

	# input_filename = "new_words/my_jp-en_short.txt"
	# model_name_en = "../modeling/model-en/W2Vmodle.bin"
	# model_name_jp = "../modeling/model-jp/W2Vmodle.bin"
	# output_filename = "output_newwords/vecs_en.csv"
	# log_filename1 = "output_newwords/Log/bad_list_en.txt"
	# log_filename2="output_newwords/Log/getVector_log.txt"

	# etrieve_vec(model_name=input_modelname_en, input_filename=input_newword_name_en, 
	# 		output_filename=output_filename, log_filename1=log_filename1, log_filename2=log_filename2)

	# model_en = Word2Vec.load(input_file2)
	# model_jp = Word2Vec.load(input_file3)
	# logout=open('./output/Log/log_en.txt','w')

	print("--- %s seconds ---" % (time.time() - start_time))
#This script could retrieve the vecoter representatiton for a give text word(pharse) lists
#basing on the word2vec model trained
#Edit 16/3/15
#Edit 16/3/17
#Normalizing the vector from a trained model so that all vector will be in length 

# coding:utf-8
import sys  
import time
start_time = time.time()
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

def nans(shape, dtype=float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a;

def vecNorm(vector):
	#vectorNorm=vector/LA.norm(vector)
	vectorNorm=vector
	return vectorNorm

logout=open('./output/Log/log_en.txt','w')
model_name='../Modeling/model-en/W2Vmodle.bin'
input_filename='./input/keyword_103_en.txt'
output_filename2=open('./output/Log/bad_list_en.txt','w')
output_filename='output/vecs_en.csv'
#fout=open('./input/goodlist_e.txt','w')

model = Word2Vec.load(model_name)
good_list=[]
counter_NaN=0
count_phrase=0
count_word=0
wnl = WordNetLemmatizer()
dim=200

with open(input_filename) as vocab_file:
	for index,vocab in enumerate(vocab_file):
		#rawVoc=vocab #keep the original un-processed word or phrase!
		#vocab=vocab.rstrip() # remove all '\n' and '\r'
		#vocab=vocab.lower()
		
		#Remove rawVoc from the raw Data is very needed!
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
				output_filename2.write(rawVoc) # no \n is needed since the 
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
						#vecP+=nans([1,dim]) 
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
				output_filename2.write(rawVoc)

	print "NaN words:",counter_NaN
	print "Pharses Number:",count_phrase
	print "all lines:",index+1
	print "all word:",count_word


	# Create a pandas Series
	df = pd.DataFrame(good_vecs,index=good_list)
	df.to_csv(output_filename,index=True,quoting=csv.QUOTE_NONNUMERIC,encoding='utf-8')

	#save the array to file
	# np.save('./output/vectors_e',good_vecs)
	# np.savetxt('./output/vectors_e.txt',good_vecs)
	#When run under the Ipython, close() is necessary preventing from 0kb issues
	# fout.close()
	output_filename2.close()
	logout.close()
print("--- %s seconds ---" % (time.time() - start_time))
#This script could do statistics for English and Japanese 

# coding:utf-8
from __future__ import division
import csv
import pandas as pd
import numpy as np
from itertools import combinations
#In order to get float result when dividing

# Global Variables
k=10
# output_dir='output/'
output_dir='output/cluster-scikit/'


def merge(k,output_dir):
	input_filename1=output_dir+'jp'+str(k)+'_good.csv'
	input_filename2=output_dir+'good_dic.csv'
	input_filename3=output_dir+'en'+str(k)+'_good.csv'

	output_filename=output_dir+'merge'+str(k)+'.csv'

	df_jp=pd.read_csv(input_filename1,header=0)
	df_dic=pd.read_csv(input_filename2,header=0)
	df_en=pd.read_csv(input_filename3,header=0)

	df=df_dic
	#df['en10']=df_en['k10']
	en10=[]
	for x in df['en']:
		#print x
		# Becareful for the poetential Unicode problem in this comparison
		result=df_en.loc[df_en['en'] == x]
		print result
		en10.append(result['k10'].iloc[0])
	#print jp10
	df['en10']=pd.Series(en10)

	jp10=[]
	for x in df['jp']:
		#print x
		# Becareful for the poetential Unicode problem in this comparison
		result=df_jp.loc[df_jp['jp'] == x]
		#print result
		jp10.append(result['k10'].iloc[0])
	#print jp10
	df['jp10']=pd.Series(jp10)

	df.to_csv(output_filename,index=False,quoting=csv.QUOTE_NONNUMERIC,encoding='utf-8')

#Get a sum of NC(number of common words) for one combination
#NOTE, DO NOT PASTE the whole function to ipython at once!
#combin is a list, NC is a tuple
def getNCSum(combin,NC):
	sum=0
	for x in combin:
		sum+=NC[x]
	#print "NC sum for combination:",combin," is ",sum
	return sum

def getNSum(combin,N):
	sum=0
	for x in combin:
		sum+=N[x]
	#print "N sum for combination:",combin," is ",sum
	return sum


if __name__ =="__main__":

	merge(k,output_dir)

	input_filename=output_dir+'merge'+str(k)+'.csv'
	df=pd.read_csv(input_filename)
	counter=[]
	for value in range(1,k+1):
		sub_df = df.ix[(df['en10']==value)]
		ser=sub_df['jp10']
		count=ser.value_counts()
		#print "for the gourp",value,"they have:"
		#print count
		counter.append(count)
	#print counter

	#The totla number of word for each JP group
	N=df.jp10.value_counts()

	max_sum=0
	for N_en in range(1,k+1):
		NC=counter[N_en-1]
		N_en_counter=df.en10.value_counts()
		N_en_count=N_en_counter[N_en]
		print "NC in group ",N_en,"is \n",NC
		NC_name_list=NC.index.values
		
		#Get the combination list
		combin_list=[]
		for i in range(1,len(NC)+1):
			combin_list+=list(combinations(NC_name_list, i))
		print "size of the combination list is in group ",N_en," is ",len(combin_list)

		#map() is not useful in this case although it is meaningful
		NC_sum_list=[getNCSum(x,NC) for x in combin_list]
		
		#PLEASE NOTE: DO NOT forget to add the number of the EN cluster (only 1)
		N_sum_list=[getNSum(x,N)+N_en_count for x in combin_list]

		sim_list=map(lambda x,y:2*x/y,NC_sum_list,N_sum_list)
		#sim_list=2*NC_sum_list/N_sum_list
		#sim_list=[2*a/b for a,b in zip(NC_sum_list,N_sum_list)]
		#print sim_list

		#Find the index of maximu similarity
		sim_bestIndex=sim_list.index(max(sim_list))
		print "the best combination similiar to EN group ",N_en,"is ",combin_list[sim_bestIndex]
		print "with the similarity of ",max(sim_list)
		print "-----------------"
	 	max_sum+=max(sim_list)
		#Create a DataFrame for print
		#df_result=pd.DataFrame()
	print "Average similairty is: ",max_sum/k



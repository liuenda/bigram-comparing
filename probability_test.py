# coding:utf-8
import pandas as pd
import numpy as np
import time
start_time = time.time()

k=10
repeat=10000
df_accuracy = pd.DataFrame(index=[False,True])
for i,x in enumerate(range(0,repeat)):
	df1=pd.DataFrame(range(0,k)).sample(k).reset_index()
	df2=pd.DataFrame(range(0,k))
	df1[1]=df2[0]
	# df1['result']=(df1[1]==df1[0])
	df_accuracy[i]=pd.DataFrame((df1[1]==df1[0]).value_counts())

print df_accuracy.sum(axis=1)
print "the accuracy is: "
print df_accuracy.sum(axis=1).iloc[1]/repeat
print "maximum prediction level: ",df_accuracy.iloc[1].max()
print("--- %s seconds ---" % 
	(time.time() - start_time))
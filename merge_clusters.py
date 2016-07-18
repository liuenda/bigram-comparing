# coding:utf-8

# 此脚本将合并通过find_vecs_en.py和find_vecs_jp.py两个脚本找出的vector来进行合并，
# 至合并两个语言共同拥有的单词，只有一方出现单词将全部被删除，所以用到了复杂的算法。

import sys  
import time
start_time = time.time()
import gensim, logging
import numpy as np
from numpy import linalg as LA
import pandas as pd
import csv
import pandas as pd

# input_filename1='output/bad_list_jp.txt'
# input_filename2='output/bad_list_en.txt'
input_filename3='output/vecs_jp.csv'
input_filename4='output/vecs_en.csv'
input_filename5='input/dic_103.csv'

output_filename1='output/good_vecs_jp.csv'
output_filename2='output/good_vecs_en.csv'
output_filename3='output/good_dic.csv'
 
# df_jp=pd.read_csv(input_filename3,index_col=0)
# df_en=pd.read_csv(input_filename4,index_col=0)
df_jp=pd.read_csv(input_filename3).rename(columns={'Unnamed: 0':'jp'})
df_en=pd.read_csv(input_filename4).rename(columns={'Unnamed: 0':'en'})
df_dic=pd.read_csv(input_filename5,names=['jp','en'])

# Be careful HERE!
# 再次逻辑错误，删除的时候两个list对应的词汇已经不同
# 因为en和jp两个list都有分别自己删除的方了
# 所以不能够通过序号来删除
# 而应该通过删除对应的词汇来删除
# 一定要想清楚逻辑！
# df_good_dic=df_dic.loc[mask_jp & mask_en] 这一步还是正确的
# 但是之后的就不对了！因为df_jp是96行，df_en是97行，不同！
mask_jp=df_dic['jp'].isin(df_jp['jp'])
mask_en=df_dic['en'].isin(df_en['en'])
df_good_dic=df_dic.loc[mask_jp & mask_en] # 想要同时加载两个mask的时候，使用&符号
df_bad_dic=df_dic.loc[~(mask_jp & mask_en)] #一定要检查列别是否正确
print "the word abonded: (MUST Check Carefully!)"
print df_bad_dic
df_jp=df_jp.loc[df_jp['jp'].isin(df_good_dic['jp'])]
df_en=df_en.loc[df_en['en'].isin(df_good_dic['en'])]

df_jp.to_csv(output_filename1,index=False,quoting=csv.QUOTE_NONNUMERIC,encoding='utf-8')
df_en.to_csv(output_filename2,index=False,quoting=csv.QUOTE_NONNUMERIC,encoding='utf-8')
df_good_dic.to_csv(output_filename3,index=False,quoting=csv.QUOTE_NONNUMERIC,encoding='utf-8')



print("--- %s seconds ---" % (time.time() - start_time))
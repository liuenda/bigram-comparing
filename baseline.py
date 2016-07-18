# coding: utf-8
import time
import pandas as pd

start_time=time.time()

input_data="new_words/my_jp-en_short.txt"
input_data2="../modeling/model-en/W2Vmodle.bin"
input_data3="../modeling/model-jp/W2Vmodle.bin"

model_en = Word2Vec.load(input_data2)
model_jp = Word2Vec.load(input_data3)
df=pd.read_table(input_data,names=["en","jp"])




print("--- %s seconds ---" % (time.time() - start_time))

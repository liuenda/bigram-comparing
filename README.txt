2016/7/18
. 添加了readme.txt
. 添加了baseline.py
   通过寻找new list里面对应的词汇
. 两个woNorm.py文件代表改脚本在进行提取word2vec表达式的时候没有进行单位化向量操作
. 运行顺序
	. find_vecs_en.py
	. find_vecs_jp.py
	. cross_check.py
	. clust_jp_en.r  通过更改k来确定cluster数
	. statistic.py   通过更改k来确定cluster数
. 添加了getVector.py
	来快速查找一个单词对应的word2vec向量
	添加了getVector.retrieve_vec() 函数
. 添加了cross_check模块的cross_check函数，可以被baseline.py随意调用


. 注意事项；
	. 在newwords中，有两种格式文件：
		. csv
		. txt
	　一定区分，在cross_check中只支持csv文件（使用了read_csv函数）
	  而在getVector时只支持txt的文件（使用了read_table函数）
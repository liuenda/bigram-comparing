2016/8/14
. 新建了基于python sci-kit learn的kmeans聚类脚本 clustering（待验证真实性）
.

2016/8/13
. 测试了baseline的可行性
. 统一了词典csv的输入格式（en,jp的header是必须的，但是顺序可以更换，比如变成jp,en）
. 修改了bug，重命名了输出的文件名称（不管是training数据还是test数据都做了全面的更换）
# 输入： a en-jp 词典, en和jp的word2vec 模型，和输出文件路径
# 输出: --> words_en.txt, words_jp.text 
    	--> log files, vecs_en.csv, vecs_jp.csv
 		--> good_vecs_en.csv, good_vecs_good_en.csv, good_dic.csv

2016/8/12 
. baseline.py 还是半成品，里面的fit_projection函数还未被使用过 （已被修改）
	新建的run()函数可以一次执行以下操作：
	. 完成jp-en词典的拆分，生成en词典1个，jp词典1个
	. 分别从en和jp的word2vec模型中抽取en词典的vector和jp词典的vector
	. 交叉检测（cross-check）删除en或者jp不包括的词汇 (原理复杂，待解释)
. run(mode) 函数中的参数可以设置成 baseline 和 proposed 两种模式，在main函数中将被逐一执行


. getVector.py
	getVector.retrieve_vec() 函数可以来快速查找一个单词对应的word2vec向量
. 创建了proposal.py来
删除了find_vecs_en.py系列的4个脚本，因为getVector.py配合baseline

. 运行顺序
	. baseline.py 
	. clust_jp_en.r  通过更改k来确定cluster数
	. statistic.py   通过更改k来确定cluster数


2016/7/18
. 添加了readme.txt
. 添加了baseline.py
   通过寻找new list里面对应的词汇
. 两个woNorm.py文件代表改脚本在进行提取word2vec表达式的时候没有进行单位化向量操作
. 运行顺序
	. find_vecs_en.py （废弃）
	. find_vecs_jp.py （废弃）
	. cross_check.py（废弃）
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
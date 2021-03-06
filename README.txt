﻿2016/8/28
. 从github中删除baseline.py
.运行顺序1(准备工作)：
	. preparation.py
	. clustering_skmeans.py (cluster_jp_en.r / spherical_kmeans.py)
	. statistic.py
. 运行顺序2：
	. experiment.py： word2word mapping，有两种方法
		..baseline()
		..proposal() 
	. articile_text.py
. 辅助脚本：
	. probability_test.py() 测试在baseline随机情况下的mapping的期望值
. 函数脚本：不直接运行，只被调用
	. cross_check.py
	. getVector.py
	. spherical_kmeans.py

2016/8/22
. 添加了top1的期望值计算
. 添加了多次试验结果求平均的Repeat()模块
. 添加了top5的期望值计算
. 记录了实验结果
. 添加了probability.py的进行概率了统计理论值计算模块


2016/8/19
. 添加了article_prediction.py

2016/8/18
. 改用spherical k-means来进行计算
	. clustering_skmeans.py spherical_kmeans.py
. 计算结果并不理想，精确度非常低
. 添加了文章级别的相似度识别模块：
步骤：
	1. 整理出100组日英文章来（暂时任意寻找）（前处理：只留下名词，可以暂时不归一）
	2. 对于日语、英语每一篇文章中的，出现的每一个单词，都转化为对应的最近的cluster编号 （1，k）
	3. 统计英语文章中的（1，k）的出现的频率，每一个文章形成一个vector（a1，a2，a3，…，a10）
	4. 用得到的mapping，将vector（a1，a2，a3，…，a10）按照mapping内容，形成一个新向量 （b1,b2,…,b10） 这个转换过程是一个固定的模式
	5. 统计英语文章中的（1，k）的出现的频率，每一个文章形成一个vector（c1，c2，c3，…，c10）
归一化以后寻找所有的vector（c1，c2，c3，…，c10）和（b1,b2,…,b10） 最接近的（相似度计算）


2016/8/17
. 调试了experiment.py的方法，更换了k的数值来检测精确度	
. 目前运行顺序
	. preparation.py
	. clustering (cluster_jp_en.r / clustering.py)
	. statistic.py
	. experiment.py
. 结果：很不理想，精度很差！

2016/8/16
. 在clustering.py 中添加 save_centroid() 函数 来保存得到的k个cluster的center
. 在experiment.py 中添加了 find_group_center() 函数 来读取计算保存mapping以后的group中心
. 在experiment.py 中完成了proposal() 函数
. 在experiment.py 中完成了baseline() 函数
分别比较了baseline和proposal方法的实验结果，表明精确度几乎没有改变（但是正确识别的单词不太一样）


2016/8/15
. 添加了基于spherical k-means cluster的skmeans方法来验证我的clustering方法的可行性
	.. 实验结果表明目前的先单位化后聚类的办法和使用skmean结果上并没有明显区别
. 目前运行顺序
	. baseline.py
	. clustering (cluster_jp_en.r / clustering.py)
	. statistic.py
	. experiment.py
    * 两个module： getVector.py 和 cross_check.py 包含独立运行代码，但是不使用
. 添加了 experiment.py 比较baseline和proposed method的实验结果
	.. 添加并且调试了baseline_method目前精度为５％左右，待检查程序的正确性和写备注。

2016/8/14
. 新建了基于python sci-kit learn的kmeans聚类脚本 clustering 
	.. 检测结果表明精度比R要低8%左右 （原因未知）
. 在statistic.py中添加了保存mapping结果的代码

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
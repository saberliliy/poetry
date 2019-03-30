# poetry
算法流程：
一．在（C:\Users\pluto\Desktop\古诗词生成\代码\数据清洗）对原始数据pre_poetry进行清洗，并且在古诗题目前后添加上”S”,”E”，作为题目的起始符和结束符，最终生成只有五言绝句组成的数据集poetry。
二．在(C:\Users\pluto\Desktop\古诗词生成\代码\word2vec)针对数据集poetry进行词向量生成，词向量维度为200维。
三．主程序：./测试/test.py
1.通过词向量，将古诗题目转换成[10,200]维的矩阵title_data（10为题目的最长长度），以及与title_data差一个时间步的矩阵target_data.将古诗词内容转换[8,1000]维的矩阵content_data。（8为古诗的行数）.
2.对content_data进行卷积处理，用以寻求句子之间的相关性。再进行一层lstm处理，保留最后状态的h和c，在利用一层lstm生成指定长度的向量，其中输入为title_data,初始状态为h和c。
3.将2的结果输入，利用全连接层，生成200维的向量，通过函数vector_to_word，生成概率最高的字。

# ATEC 花呗金融大脑NLP比赛
## 赛题简述
给一对给花呗客服的提问，判断是否表达了一个意思。
该问题属于NLP领域的 Paraphrase Identification 问题，和 Text Similarity 问题有一定关联。
数据集中完全相同的语句只出现一次。
## 项目组织简述

- README.md 说明文档
- input/ 存放输入的文件夹    
  - data_tr.csv,data_te.csv (训练和测试数据，考虑到版权没有放出，请去[官网](https://dc.cloud.alipay.com/index#/topic/intro?id=3)下载，本地要求格式为id,label,sent1,sent2
  - w2v.csv 用到的词向量，格式是第一列为词，其余列为数字，英文逗号分割
- input_online.py 模拟线上训练输入环境
- input_predict.py 模拟线上测试环境
- ensemble.py 集成训练的文件
- ensemble_predict.py 模型预测输出的文件
- models/dict.txt 自定义词典
- Dockerfile 项目仿真使用的docker虚拟机
- requirements.txt Docker虚拟机的环境。(tensorflow未列出)

## 数据预处理
没有进行数据预处理。集成的时候使用了词级别模型和字级别模型。序列均填充到长为20. 使用了来自[苏剑林博客](https://kexue.fm/archives/4304)的词向量
没有使用传统数据挖掘特征。
## 模型简述
1. 模型用python3.5 + keras实现。keras版本为2.1.6，tensorflow 版本为1.5.0。输入层有spatial dropout. 编码层为两层双向 128单元lstm，共享权重。池化层使用了MaxPooling 和AveragePooling。交叉层 为 $ f (x_1,x_2)=(|x1-x2|,-x1.*x2) $，
FC层为两层Relu全连接层。用10折验证训练模型的时候对模型结构做了扰动，其中4个模型使用词向量做嵌入层，其余使用字向量。

2. 测试时中使用了半监督学习。 选取集成模型输出概率值 $ <0.25 $或 $ >0.75 $ 的样本加入训练集中，将成对的测试集序列重新组对作为负类加入训练集中，
训练5个epoch。本地验证可看见auc微弱上升，f1提升不明显。 在比赛中由于提交次数过少实际效果不明 :) 

## 参考资料
1. [魔镜杯第7的解决方案](https://qrfaction.github.io/2018/07/25/%E9%AD%94%E9%95%9C%E6%9D%AF%E6%AF%94%E8%B5%9B%E7%AD%94%E8%BE%A9PPT/)
2. [github上开源的语义匹配模型库](https://github.com/faneshion/MatchZoo)
3. [句子对建模的综述](https://arxiv.org/abs/1806.04330)
4. [stanford一个数据库和大量模型的性能](https://nlp.stanford.edu/projects/snli/)
## 个人感受
1. 在处理口语化的短文本时， 字级别向量比词级别可能更好
2. 时间维度上的MaxPooling比Attention更适合这个问题，KMaxPooling没有显著优于MaxPooling和AveragePooling。
3. [孪生网络](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf)性能远差于编码+交互+FC的模型。可能是因为该数据集上的同义关系由于噪音+争议数据并不完全满足自反对称传递性质加上孪生网络的表达能力有限。
4. 双向RNN性能十分适合此任务。编码层全部或者部分换为transformer（这一点和排名靠前的某个队伍不一致，可能是我实现有问题）,CNN,或是用基于交互的CNN，基于交互的HAN性能都劣于双向RNN。
5. 初赛时使用了[基于谷歌翻译的增广](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/48038)，发现没有用。。花光了谷歌云的赠金还差点玩脱
6. 尝试过一些数据增强手段，包括在词向量空间进行[mixup](https://arxiv.org/abs/1710.09412) ,[随机反转一些样本](https://arxiv.org/abs/1605.00055)，没有明显改善。
## 没有尝试的思路
1. 词向量和字向量的结合提升了性能，如果使用拼音序列能否进一步提升性能呢？由于平台环境限制没有尝试。

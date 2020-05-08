# garbage_calssify-by-resnet50-

#### 参考项目 ：https://github.com/zrongcheng/huaweicloud_garbage_classify   
根据该项目修改了预处理方式，并用了其中的一些小tricks：比如label smooth，random crop等。测试的地方使用了random crop，然后集成。

#### 项目描述
1. 任务是对垃圾图片进行分类，即首先识别出垃圾图片中物品的类别（比如易拉罐、果皮等），然后查询垃圾分类规则，输出该垃圾图片中物品属于可回收物、厨余垃圾、有害垃圾和其他垃圾中的哪一种。垃圾类别共40类。
2. [训练数据集下载地址](https://competition.huaweicloud.com/information/1000007620/introduction?track=107)

#### 准确率
1. 在官方测试集上有0.893513。
2. 在网站上收集了每类20张左右的垃圾图片，共40类，800张。在该测试集上，模型准确率有0.86125。测试集名字是test_data。


#### 改进过程

模型名称 | 主要内容 | 训练轮次 | 得分 
---|--- | --- | ---
官方baseline | Resnet50 | 50|67
model1 | 数据集进行图片增强，添加dropout |50| 77
model2 | 在model1的基础上，增加迁移学习和微调 | 3 | 71
model3 | 在model2的基础上，修改预处理方式(采用resnet50自带的) | 5 | 85
model4 | 在model3的基础上，进行图片增广，每类1000张左右 | 5 | 85
model5 | 在model4的基础上修改预处理方式(resize成（ 256,256))，并将图片增强和图片遮挡写入代码 | 5 | 89

#### 总结

在该项目中，主要使用迁移学习将resnet50用于垃圾分类，迁移学习可以大大地节约训练时间，并在短时间内达到更高的准确率。

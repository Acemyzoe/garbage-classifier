# garbage calssify used resnet50

#### 参考项目 ：https://github.com/zrongcheng/huaweicloud_garbage_classify   
一些小tricks：比如label smooth，random crop等。

## 比赛描述
1.   本赛题采用深圳市垃圾分类标准，赛题任务是对垃圾图片进行分类，即首先识别出垃圾图片中物品的类别（比如易拉罐、果皮等），然后查询垃圾分类规则，输出该垃圾图片中物品属于可回收物、厨余垃圾、有害垃圾和其他垃圾中的哪一种。  
2. 垃圾分类标准：
   - 可回收物指适宜回收和资源利用的废弃物，包括废弃的玻璃、金属、塑料、纸类、织物、家具、电器电子产品和年花年桔等。
   - 厨余垃圾指家庭、个人产生的易腐性垃圾，包括剩菜、剩饭、菜叶、果皮、蛋壳、茶渣、汤渣、骨头、废弃食物以及厨房下脚料等。
   - 有害垃圾指对人体健康或者自然环境造成直接或者潜在危害且应当专门处理的废弃物，包括废电池、废荧光灯管等。
   - 其他垃圾指除以上三类垃圾之外的其他生活垃圾，比如纸尿裤、尘土、烟头、一次性快餐盒、破损花盆及碗碟、墙纸等。
3. [训练数据集下载地址](https://competition.huaweicloud.com/information/1000007620/introduction?track=107)

## use

`python run_model.py`

函数描述：

```python
# train
train_model('./garbage_data/train_data/')

# validation
test_single_h5('./garbage_model.h5','./garbage_data/test_data/')
```


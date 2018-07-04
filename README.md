# WA-ModelEnsemble
Weight Averaging Model Ensemble

> 模型融合的方法很多，Voting、Averaging、Bagging 、Boosting、 Stacking，那么一些kaggle比赛中选手会选用各种方法进行融合，其中岭回归就是一类轻巧且非常有效的方法，当然现在还有很多更有逼格的方法。本文是受快照集成的启发，把[titu1994/Snapshot-Ensembles](https://github.com/titu1994/Snapshot-Ensembles)项目中，比较有意思的加权平均集成的内容抽取出来，单独应用。


blog链接：https://blog.csdn.net/sinat_26917383/article/details/80905004

code：```machineLearning14.ipynb```是主要实现；```MinimiseOptimize.py```是最小优化函数。


## 1、 快照集成

因为受其启发，所以在这提一下，快照集成是一种无需额外训练代价的多神经网络集成方法。 通过使单个神经网络沿它的优化路径进行多个局部最小化，保存模型参数。 利用多重学习速率退火循环实现了重复的快速收敛。

![这里写图片描述](https://github.com/mattzheng/WA-ModelEnsemble/blob/master/pic/001.jpg)

![这里写图片描述](https://github.com/mattzheng/WA-ModelEnsemble/blob/master/pic/002.jpg)


### 1.1 比较有意思的做法

作者在训练相同网络时使用权重快照，在训练结束后用这些结构相同但权重不同的模型创建一个集成模型。这种方法使测试集效果提升，而且这也是一种非常简单的方法，因为你只需要训练一次模型，将每一时刻的权重保存下来就可以了。

也就是，同一款模型，在学习率稍微调高，**训练中得到的不同阶段的模型文件都保存并拿来做最后的模型融合**。

**长学习率循环的思想** 在于能够在权重空间找到足够多不同的模型。如果模型相似度太高，集合中各网络的预测就会太接近，而体现不出集成带来的好处。


### 1.2 权重的解决方案

对于一个给定的网络结构，每一种不同的权重组合将得到不同的模型。因为所有模型结构都有无限多种权重组合，**所以将有无限多种组合方法。**

训练神经网络的目标是找到一个特别的解决方案（权重空间中的点），从而使训练集和测试集上的损失函数的值达到很小。

### 1.3 相关实现：cifar100 图像分类任务

可参考项目：[titu1994/Snapshot-Ensembles](https://github.com/titu1994/Snapshot-Ensembles)

该项目用keras1.1 做了cifar_10、cifar_100两套练习，使用的是比较有意思的图像框架： Wide Residual Net (16-4)。作者已经预先给定了5款训练快照，拿着5套模型的预测结果做模型集成，使使训练集和测试集上的损失函数的值达到很小。


----------


## 2、 14款常规的机器学习模型
sklearn官方案例中就有非常多的机器学习算法示例，本着实验的精神笔者借鉴了其中几个。本案例中使用到的算法主要分为两套：

 - 第一套，8款比较常见的机器学习算法，```"Nearest Neighbors", "Linear SVM", "RBF SVM",
            "Decision Tree", "Neural Net", "AdaBoost",         "Naive Bayes", "QDA‘’```（参考：[Classifier
   comparison](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py)）
 - 第二套，偏向组合方案，`RandomTreesEmbedding, RandomForestClassifier,
   GradientBoostingClassifier、LogisticRegression`（参考：[Feature
   transformations with ensembles of
   trees](http://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#sphx-glr-auto-examples-ensemble-plot-feature-transformation-py)）

![这里写图片描述](https://github.com/mattzheng/WA-ModelEnsemble/blob/master/pic/003.jpg)

机器学习模型除了预测还有重要的特征筛选的功能，不同的模型也有不同的重要性输出：

### 2.1 特征选择

在本次10+机器学习案例之中，可以看到，可以输出重要性的模型有： 

 - 随机森林`rf.feature_importances_`
 - GBT`grd.feature_importances_` 
 - Decision `Tree decision.feature_importances_` 
 - AdaBoost `AdaBoost.feature_importances_`
 
可以计算系数的有：线性模型，`lm.coef_` 、 SVM `svm.coef_`

Naive Bayes得到的是：NaiveBayes.sigma_`解释为：variance of each feature per class`

### 2.2 机器学习算法输出

算法输出主要有：重要指标（本案例中提到的是`acc/recall`）、ROC值的计算与plot、校准曲线（Calibration curves）

![这里写图片描述](https://github.com/mattzheng/WA-ModelEnsemble/blob/master/pic/004.jpg)

该图为校准曲线（Calibration curves），Calibration curves may also be referred to as reliability diagrams. 是一种算法可靠性检验的方式。

.

----------


## 3、optimize 权重空间优化

主要是从[titu1994/Snapshot-Ensembles](https://github.com/titu1994/Snapshot-Ensembles)抽取出来的。简单看看逻辑:

### 3.1 简述权重空间优化逻辑

##### 3.1.1 先定义loss函数：
```
# Create the loss metric 
def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = np.zeros((sample_N, nb_classes), dtype='float32')

    for weight, prediction in zip(weights, preds):
        final_prediction += weight * prediction

    return log_loss(testY_cat, final_prediction)
```
testY_cat为正确预测标签， final_prediction为多款模型预测概率组合。

##### 3.1.2 迭代策略
```
minimize(log_loss_func, prediction_weights, method='SLSQP', bounds=bounds, constraints=constraints)
```

SciPy的optimize模块提供了许多数值优化算法，minimize就是其中一种。

### 3.2 实践

具体code笔者会上传至笔者的github之上了。

步骤为：

 - 1、随机准备数据`make_classification`
 - 2、两套模型的训练与基本信息准备
 - 3、观察14套模型的准确率与召回率
 - 4、刻画14套模型的calibration plots校准曲线
 - 5、14套模型的重要性输出
 - 6、14套模型的ROC值计算与plot
 - 7、加权模型融合数据准备
 - 8、基准优化策略：14套模型融合——平均
 - 9、加权平均优化策略：14套模型融合——加权平均优化

一些细节了解：

#### 3.2.7  加权模型融合数据准备
```
# 集成数据准备
preds_dict = {}
for pred_tmp,name in [[predictEight[n]['prob_pos'],n] for n in names] + [(y_pred_lm,'LM'),
                       (y_pred_rt,'RT + LM'),
                       (y_pred_rf_lm,'RF + LM'),
                       (y_pred_grd_lm,'GBT + LM'),
                       (y_pred_grd,'GBT'),
                       (y_pred_rf,'RF')]:
    preds_dict[name] = np.array([1 - pred_tmp , pred_tmp]).T

# 参数准备
preds = list(preds_dict.values())
models_filenames = list(preds_dict.keys())
sample_N,nb_classes = preds[0].shape
testY = y_test.reshape((len(y_test),1))  # 真实Label (2000,1)
testY_cat = np.array([1 - y_test ,y_test]).T # (2000,2)   
```

models_filenames 代表模型的名字；sample_N样本个数；nb_classes 分类个数（此时为2分类）；testY 真实label；testY_cat 基于真实Label简单处理。


#### 3.2.8  基准优化策略：14套模型融合——平均

```
def calculate_weighted_accuracy(prediction_weights):
    weighted_predictions = np.zeros((sample_N, nb_classes), dtype='float32')
    for weight, prediction in zip(prediction_weights, preds):
        weighted_predictions += weight * prediction
    yPred = np.argmax(weighted_predictions, axis=1)
    yTrue = testY
    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    recall = recall_score(yTrue, yPred)
    print("Accuracy : ", accuracy)
    print("Recall : ", recall)

# 模型集成：无权重
    # 无权重则代表权重为平均值
prediction_weights = [1. / len(models_filenames)] * len(models_filenames)
calculate_weighted_accuracy(prediction_weights)
>>> Accuracy :  79.7
>>> Recall :  0.7043390514631686
```

对14套模型，平均权重并进行加权。可以看到结论非常差。

#### 3.2.9  加权平均优化策略：14套模型融合——加权平均优化

```
def MinimiseOptimize(preds,models_filenames,nb_classes,sample_N,testY,NUM_TESTS = 20):
    best_acc = 0.0
    best_weights = None
    # Parameters for optimization
    constraints = ({'type': 'eq', 'fun':lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * len(preds)

    # Check for NUM_TESTS times
    for iteration in range(NUM_TESTS):  # NUM_TESTS,迭代次数为25
        # Random initialization of weights
        prediction_weights = np.random.random(len(models_filenames))

        # Minimise the loss 
        result = minimize(log_loss_func, prediction_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        weights = result['x']
        weighted_predictions = np.zeros((sample_N, nb_classes), dtype='float32')

        # Calculate weighted predictions
        for weight, prediction in zip(weights, preds):
            weighted_predictions += weight * prediction

        yPred = np.argmax(weighted_predictions, axis=1)
        yTrue = testY

        # Calculate weight prediction accuracy
        accuracy = metrics.accuracy_score(yTrue, yPred) * 100
        recall = recall_score(yTrue, yPred)

        print('\n ------- Iteration : %d  - acc: %s  - rec:%s -------  '%((iteration + 1),accuracy,recall))
        print('    Best Ensemble Weights: \n',result['x'])
        
        # Save current best weights 
        if accuracy > best_acc:
            best_acc = accuracy
            best_weights = weights
    return best_acc,best_weights

# 模型集成：附权重
best_acc,best_weights = MinimiseOptimize(preds,models_filenames,nb_classes,sample_N,testY,NUM_TESTS = 20)

>>> Best Accuracy :  90.4
>>> Best Weights :  [1.57919854e-02 2.25437178e-02 1.60078948e-01 1.37993631e-01
	 1.60363024e-03 1.91105368e-01 2.34578651e-02 1.24696769e-02
	 3.18793907e-03 1.29753377e-02 1.12151337e-01 7.62845967e-04
	 3.05643629e-01 2.34089531e-04]
>>> Accuracy :  90.4
>>> Recall :  0.9112008072653885
```

在迭代了20次之后，通过加权求得的综合预测水平，要高于平均水平很多。不过，跟一些比较出众的机器学习模型差异不大。






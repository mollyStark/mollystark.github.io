title: "faster-rcnn源码解读－RPN"
date: 2019-01-21 17:18:47
tags: [faster-rcnn, rpn]
math: true
---

faster-rcnn是经典的图片检测模型，在这个模型中，首次提出了RPN(region proposal
network)这个网络结构，用来生成候选区域。这个任务之前使用一些独立检测方案，RPN
首次使用神经网络去训练，从而使整个检测方案是端到端(end-to-end)的实现方式。这篇
文章主要解读了faster-rcnn中RPN相关的代码。

## RPN网络结构
RPN网络是用来生成候选框的网络，前承CNN网络的特征抽取部分，后接更具体的分类和回
归模型，输入是CNN卷积层抽取的图片高维特征，输出是n个正例检测框和m个负例检测框
。

我们可以结合`py-faster-rcnn`中的prototxt文件画出faster-rcnn网络的整体结果图和
具体的rpn的结构图。

{% asset_img rpn-Page-1.png faster-rcnn网络结构图 %}
faster-rcnn整体网络结构

{% asset_img rpn-Page-2.png RPN网络结构图 %}
RPN网络结构

在卷积网络之后，RPN网络首先是又用了两层卷积网络，得到前背景二分类的分类结果和
坐标回归结果，输出大小分别是2x9 , 4x9 ，9代表了每个框anchor的种类数（下文会讲
到），2是分类数（0/1），4是坐标数（x1, y1, x2, y2）。

然后连接了`anchor_target_layer`，筛选和groundTruth匹配的anchors样本，然后后面
是softmaxWithLoss层和smoothL1Loss层计算这些anchors的分类和回归损失，以学习和优
化RPN的提取。

另一方面，在RPN的卷积层后连接了`proposal_layer`层，根据前面分类的结果筛选分类
样本，然后连接`proposal_target_layer`层，生成训练样本和训练目标，接到后面的fc
层等优化loss。

这几个层都是用python接口写的，下面，具体结合代码看一下。

## RPN网络代码解析

在faster-rcnn源码中，rpn网络相关的代码在lib/rpn目录下面，主要有这几个文件：
1. generate_anchors.py：生成不同尺度不同大小的anchors。
2. anchor_target_layer.py：生成每个anchor的分类标签（0/1）和回归坐标。
3. proposal_layer.py：根据anchor的分类结果生成候选框集合（包括每个anchor的分类
   值和回归的坐标）。
4. proposal_target_layer.py：生成每个候选框的分类标签（0-k）和回归坐标。
5. generate.py：根据训练好的RPN网络生成检测候选集。

这里面有两个概念，anchor和proposal，proposal很好理解，就是筛选出来的候选框，而
anchor，是RPN网络的一个先验结构，可以认为是候选框有多大的限定。对于一张图片，
RPN 认为需要检测的物体具有某些特定的长宽比，对这些特定的长宽比，又可以根据不同
的尺度，即分辨率，得到不同的检测框，这些预先定好的检测框就被称为anchors，对这
些anchors指定的区域特征进行分类和筛选，就得到proposals。

下面，我们分别来看这些代码，最后梳理出整个RPN的逻辑。

### generate_anchors.py

这个文件用来生成不同大小不同分辨率的anchor框，主要的函数是`generate_anchors`。
看一下它的代码：
```
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors
```
生成的步骤包括：
1. 生成一个左上坐标为(0, 0), 右下坐标为(15, 15) 的`base_anchor`
2. 基于`base_anchor`，用`_ratio_enum`方法生成不同尺寸的anchors，默认ratios是
   [0.5, 1, 2]，即3种尺寸的ratio，不同尺寸的生成框面积都是16x16，但长宽比不一
   样。
3. 对上面每个尺寸的框，用`_scale_enum`方法再生成不同尺度的anchors，默认scales
   是[8, 16, 32]，不同scale生成的框长宽比不变，但是长宽绝对值变大相应的倍数。
4. 将这些生成框排列起来（np.vstack()操作）。

其中，`_ratio_enum`和`_scale_enum`中都有生成anchor的操作，具体方法是
`_mkanchors`函数，根据中心点和长宽生成左上点和右下点的坐标。

### anchor_target_layer.py 
这个文件是一个类文件，继承自caffe的Layer类，caffe layer（后续简称caffe层）包括
`setup`、`forward`和`backward`函数，其中`setup`方法用来初始化一些参数，
`forward`方法定义根据输入生成输出的方法，是这一层操作的主要函数，其输出可以用
做下一个caffe层的输入，而`backward`方法定义在反向传播时如何更新参数。

`anchor_target_layer.py`目标是筛选生成的anchor，得到样本的分类标签（0/1）和标
签为1的bbox的学习项，为后面计算loss提供数据。

首先`setup`函数将anchors坐标生成出来，`forward`函数输入是标注的框和图片信息，
输出是筛选后的anchor，以及anchor的训练坐标和标签。

具体的方法做了这些事情：
1. 根据不同的anchors生成候选框，首先枚举了rpn分类featureMap中所有的坐标点，生
   成了对应原图尺寸的K*A个anchor，K是最后分类的个数，A是anchor的种类数目，即上
   面所说的3（不同ratio）*3（不同scale）=9个，然后保存在图像边界内的anchors。
2. 计算出这些anchors和gt框的重叠，计算公式是 
3. 根据重叠筛选正负样本，即前背景的二分类样本，挑选规则如下：
    1. 对每个ground truth，IoU值最大的anchor，标记为正样本，label=1
    2. 如果anchor box与ground truth的IoU大于某阈值，标记为正样本，label=1
    3. 如果anchor box与ground truth的IoU小于某阈值，标记为负样本，label=0
    4. 正负样本如果超过一定量，则做下采样。
4. 在挑选出正负样本后，计算回归的坐标差值，并对样本设置了权重，即
   `bbox_inside_weights`和`bbox_outside_weights`。
5. 把采样的样本映射到原来的样本顺序(`_unmap`)，并填充`top`数据。

### proposal_layer.py

`proposalLayer`也继承自caffe层，目标是根据分类结果生成候选框集合作为输出，
`setup`函数和上面的`anchor_target_layer`类似，也是将anchors坐标生成出来，
`forward`函数输入是rpn分类和回归的输出，以及原始图片信息，包含在`bottom`中，输
出包含在`top`中，`top[0]`是筛选出的proposal的roi数据，`top[1]`是筛选出的
proposal的分数。

具体的方法做了这些事情：
1. 根据不同的anchors生成候选框，首先枚举了所有的点，生成了K*A个anchor，K是最后
   分类的个数，A是anchor的种类数目，即上面所说的3（不同ratio）*3（不同scale）
   =9个，最后把bbox delta 和分类score都reshape到一样的维度。
2. `bbox_transform_inv`函数根据anchor坐标和bbox delta生成proposal候选集坐标。
3. `clip_boxes`函数修正超出图片大小的候选集的坐标。
4. `_filter_boxes`把面积小于某个阈值的候选框筛选出去。
5. 根据分类得分对候选框排序，得到分数最高的n个候选框。
6. 应用nms(非极大值抑制)算法，得到分数最高的n个候选框。
7. 将最后的候选框封装到`top`输出给下一层。


### proposal_target_layer.py

`proposal_target_layer.py`也继承自caffe层，目标是为生成的proposal匹配分类标签
(1-K) 和bbox坐标学习项。

`forward`函数输入是`proposal_layer`的输出（即候选框区域），和要学习的标注框，
输出是用作训练样本的标注框和相关的学习目标（坐标差值和分类类别）。

在`proposal_layer`后拿到待训练的候选框之后，需要给候选框分配相应的标注，因此
，主要的函数就是`_sample_rois`这个函数。

`_sample_rios`函数步骤：
1. 计算rois和gt_boxes的重叠区域。
2. 挑选重叠区域大于FG_THRESH的前景并随机挑选一些作为正样本
4. 挑选重叠区域小于BG_THRESH_HI且大于BG_THRESH_LO的背景并随机挑选一些作为负样
   本。
5. 计算挑出来的样本的bbox_data和label。

### generate.py
这个文件中有两个主要的函数`im_proposals`和`imdb_proposals`，分别是将图片数据和
imdb数据传送到RPN网络得到候选集输出。这里就不详细展开了。

## 总结
根据rpn目录下的所有文件以及网络结构文件，我们再总结下rpn的框架脉络。rpn网络用
来训练检测框的分类和提取，分别是用一个两层的CNN（共享第一层卷积）来做抽取，生
成的维度是一个超参，也就是anchors的数量，loss是用`rpn_target_layer`筛选的样本
和背景标注进行比对来计算。

rpn网络的分类和回归结果同时用来生成候选集proposals（`proposal_layer`和
`proposal_target_layer`），然后用`ROIPooling`生成ROIs，对每一个ROI区域做了更高
维的特征提取，用来做最终的分类检测和回归检测。


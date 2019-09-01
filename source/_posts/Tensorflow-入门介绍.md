title: Tensorflow-入门介绍
date: 2019-08-10 10:32:43
tags: Tensorflow
---

Tensorflow现在是最火热的深度学习框架之一，但是对第一次使用的人来说，里面的一些
概念和用法真的是太不友好了，因此这篇文章就介绍一些基础概念，希望給新手入门一些帮助。

# Tensorflow计算图(Graph)
Tensorflow计算图是Tensorflow中最重要的一个概念，没有之一。计算图是对计算过程的
一个抽象。通常我们在python中做计算的方法很简单，比如说做一个加法，就是：
```
a = 1
b = 2
c = a + b
print(c)
```
这是一个一步步执行的计算过程，最后我们print出一个结果3。
但是在Tensorflow中，做一个简单的加法需要两个步骤，首先是定义一个计算图，定义完
以后通过某些方式才去执行（下文讲）。我们先来看一下实现：
```
import tensorflow as tf
a = tf.constanst(1)
b = tf.constanst(2)
c = tf.add(a, b)
print(c)

with tf.Session() as sess:
    print(sess.run(c))
```
看代码，第一段代码和python的基本逻辑好像差不多，但是后面加上了一个对象做运行的操作，emmm，事情肯定没这么简单。运行一下，你会发现，打印出来的是这样的：
```
Tensor("Add:0", shape=(), dtype=int32)
3
```
果然，第一次print并没有直接出来结果，第二次print才打印出来我们想要的计算结果3。
仔细看第一次的打印结果，是一个表示加法操作的`Tensor`对象，这是因为在Tensorflow中，我们上面的代码操作实际上是在默认的计算图中生成了一些对象，还没有真正的执行计算。打印出来的也只是生成的对象的一些信息。
第一次接触这种方式会觉得很奇怪，为什么要用这种方式呢？
其实这是两种不同的编程思路，python那种是命令式编程(imperative style programs)，
tensorflow这种是符号式编程(symbolic style programs)。这两种方式的区别在于，
tensorflow在定义计算时并没有真正去做数值计算，定义好后后面隐式地做了编译，最
后再得出结果。
有这么几种原因，导致用计算图的方式会更好：
1. 采用定义然后编译运行的方式，有利于在编译时做一些优化。python的方式每一行
   都会要去执行，但是tensorflow的方式由于预先知道了整个计算流程，因此可以做一些优化，比如优化一些不必要的执行，或者做一些并行处理。
2. 由于通常tensorflow会在大数据集上去做计算，实际上是调用更底层的语言，如C语言
   ，cuda加速等等去进行高效的计算的，使用先编译的方式能让计算的流程串在一起，
   不像numpy一样，要不停地做numpy对象和python对象的转换。
3. 由于tensorflow通常是做的深度学习相关的计算，包括会有反向传播等的流程，使用
   图的方式更助于计算的复用。

在一个文件中，可以指定多个计算图，分别执行计算，互不干扰，比如说一个程序里面
需要进行不同的数据处理，融合模型，那么此时就需要创建多个计算图。
说到图的话，我们肯定会遇到两个概念，边和节点。在Tensorflow中，边是数据，在
Tensorflow中定义为Tensor，节点包
括输入节点，输出节点和运算节点。Tensor根据计算图在节点中流动(flow)，这就是Tensorflow啦。

# Tensor
Tensor的意思是张量，是比矩阵更高阶的一种数据结构，尽管在Tensorflow中，
Tensor是节点的输入或输出，在计算图中流动，但它实际上并不直接是数据本身，而是一个指向数据的抽象，背后还有一系列的信息，包括数据在哪里存储，属于哪个计算图，生
成这些数据的操作等等。只有当它被传入session中时，数据才真正被计算出来。

# Session
Session（会话）是一个字面意思很不好理解的概念，通常会话指的是一个交互环境，这
里我觉得Session可以认为是执行操作和计算数据的环境。
通常我们使用
```
with tf.Session() as sess:
    sess.run(xxx)
```
来执行计算，使用with语句是因为Session里会包括一些创建的变量，队列等等资源，所以需要最后
将资源释放。
要执行计算获得结果，需要执行session.run()方法，这个方法定义如下：
```
run(
    fetches,
    feed_dict=None,
    options=None,
    run_metadata=None
)
```
运行时会根据计算图，把从输入到fetches里每一个节点的路径上的操作都作为子图计算，输出是fetches里
每一个对象的值。这里需要注意的是，如果session.run()计算一个列表，那列表中的
每一个节点都存在一个计算路径，但是这些路径的公共路径只会计算一次，而如果是分开每
一个fetch做一次session.run()的话，就会有重复的计算被多次计算。
fetches里的变量执行的顺序其实是不确定的，因为编译时是生成了计算图，执行时除非
两个变量在一个计算子图中，存在严格的顺序关系，否则由于并行计算的原因，实际上输
出的值是无法预测的。
比如这样一个例子
```
import tensorflow as tf
x = tf.Variable(1)
op = tf.assign(x, x + 1)
x = x + 0                   # 1

with tf.Session() as sess:
  tf.global_variables_initializer().run()
  for i in range(5):
    print(sess.run([x, op]))
```
x是一个变量，x+1是一个Tensor，依赖于x，tf.assign也是一个Tensor，依赖于x+1，op的值就是x+1的值，同时运行时x的值会更新为x+1的值。因此在计算图中，op始终是
在x的后面的。而#1这行，x重新定义为一个Tensor，指向x+0。这个对象同样依赖变量x，但是跟op完全没有关系。
如果我们画一张变量图，应该是这样的：

{% asset_img Tensorflow.jpg Tensorflow变量图 %}
变量示意图

在sess.run中，参数里面的x其实是后面的Tensor x,而不是最初的变量x，也就是图中右边蓝色的那个'+'节点，而op指向'assign'节点，在运行时，op
和x并没有强依赖关系，因此每次跑出来的不是完全的+1关系。
某一次的运行结果：
```
[1, 2]
[3, 3]
[3, 4]
[5, 5]
[6, 6]
```
如果没有#1这行，那么op和x在一个计算子图中，是强依赖关系。每次运行sess.run()的时候，op和x都是相等的。
计算结果：
```
[2, 2]
[3, 3]
[4, 4]
[5, 5]
[6, 6]
```
那么，如果我们想让x=x+0执行在op之前的话，需要这样定义这段代码：
```
import tensorflow as tf
x = tf.Variable(1)
x2 = x + 0                   # 1
with tf.control_dependencies([x]):
    op = tf.assign(x, x2 + 1)

with tf.Session() as sess:
  tf.global_variables_initializer().run()
  for i in range(5):
    print(sess.run([x, op]))
```


# 总结
我们初步了解了Tensorflow的结构，它是一种符号式编程语言，首先定义计算图，在
session.run()的时候才真正去计算数据。这样的设计给调试带来了很多不便，但是带来
了计算上的优化空间。
同样是符号式的caffe，其实也定义了模型，但是和Tensorflow不
一样的是，caffe的对象更加粗粒度，最细的粒度是层，定义了层的前向和反向操作，相对来说不利于扩展，如果要实践一些新的想法，哪怕是非常细微的改变，比如从单标签分类到多标签分类，都需要重新定义层，而
Tensorflow的定义更加细粒度，是细到加法运算的，我们能更快的做一些新的尝试。

# 参考文献
[1] https://zhuanlan.zhihu.com/p/38812133

[2] https://www.tensorflow.org/guide/extend/architecture

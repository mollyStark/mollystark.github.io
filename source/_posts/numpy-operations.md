title: numpy operations
date: 2019-01-18 17:34:34
tags:
math: true
---

Numpy 库是python中进行科学计算的基础包，如果用python去进行矩阵运算，肯定需要
用到numpy库，最近在看faster-rcnn的源码，看到了很多numpy矩阵运算的操作，结合
numpy的manual，这里做一个关于numpy运算的不定期记录和总结。

## numpy 的核心－ndarray 对象
ndarray 是numpy里的核心类，封装了对$n$ 维矩阵的各种编译好的操作和方法。ndarray
数组和python list的区别：
1. ndarray中的元素必须在创建时指定大小，而python list是可以动态增长的。
2. ndarray中的元素必须属于相同的类型，而python list中的元素不需要。
3. ndarray进行大量数据的计算比list效率更高（接近c程序的速度）。
4. 现在越来越多的库使用numpy作为底层的计算库。

举个例子，如果要计算矩阵 $A$ 的元素和矩阵 $B$ 的元素相乘，那么在python里，我
们需要这样写：
```
C = []
for i in range(len(A)):
    C[i] = A[i] * B[i]
```
而使用 numpy 库，我们只需要一句话：
```
C = A * B
```
是不是很简洁，就像是写数学公式一样。

当然上面语句可以运行的一个最简单的前提是：A和B的大小是一样的。在简洁的向量化
的语法背后，其实有一些约定好的机制，定义了怎样的运算是被允许的，怎样是不被允许的。

## Broadcasting机制
在numpy库中，**所有**的运算，包括代数计算，逻辑运算，位运算等等，都是隐式转换
成一个元素一个元素的运算的。这个就是Broadcasting机制。所以，上面的运算可以成
立，前提是要满足可以隐式扩展成一个元素对一个元素的方式。具体可以用这样两条匹
配规则概括：
两个矩阵，从最后一维开始往前推，
1. 两个维度相等，或者
2. 某一个矩阵的这一维维度是1.

对于两个矩阵的某一个维度的元素来说，若维度相等，那么元素是可以一一对应的，若维度不相等，但是有一个矩
阵维度是1，那么维度是1的那个可以通过复制的方式扩展到另一个矩阵的相应维度，从
而达到元素一一对应的效果。

举个例子，
```
>>> import numpy as np
>>> x = np.array([[1, 2, 3], [4, 5, 6]])
>>> y = np.array([7, 8, 9])
>>> x + y
array([[ 8, 10, 12],
       [11, 13, 15]])
```
x是一个大小为（2， 3）的矩阵，而y的大小是（1， 3），从后往前看维度信息，最后一
维都是3，是满足条件的，然后第一维y的维度是1，因此也满足条件，最后加法的效果就
相当于把y数组拷贝了一行[7, 8, 9]与x相加。

当然，实际上，numpy并没有真的进行数组的拷贝，否则数组一大，就会有空间上的巨大开销，
上面所说的复制的方式都是一种便于理解的说法，而numpy其实是用了一种记录strides的trick
去扩展数组而不过度的占用存储空间的。

底层实现是在nditer类用c代码实现的，但是我们可以在python中通过 `np.lib.stride_tricks` 里的方法来看一下是
怎么进行broadcast的。
```
# 得到broadcast后的数组A和B
>>> A,B=np.lib.stride_tricks.broadcast_arrays(np.arange(6).reshape(2,3),
...                                       np.array([[1],[2]]))
>>> A
array([[0, 1, 2],
       [3, 4, 5]])
>>> B
array([[1, 1, 1],
       [2, 2, 2]])

# A的shape和strides, 说明数组A从第一维的第一个元素到第一维的第二个元素需要跳过24bytes的offset
，也就是第二维的元素个数*每个元素的大小（3*8），而从第二维的第一个元素到第二维的第二个元素需要
跳过8bytes的offset，也就是一个元素的大小。
>>> A.shape, A.strides
((2, 3), (24, 8))

# B的shape和strides，说明数组B从第一维的第一个元素到第一维的第二个元素需要跳过8bytes的offset
，也就是第二维的真实元素个数1*每个元素的大小8bytes，而从第二维的第一个元素到第二维的第二个元素
没有offset，就是还是这个元素，说明底层numpy并没有真的将数组进行拷贝复制。
>>> B.shape, B.strides
((2, 3), (8, 0))
```
上面的例子中，一个元素占用的内存是8字节，B虽然也是shape为(2, 3)的数组，但是它
的strides是(8, 0)，通过shape和strides的组合，数组可以通过一种“虚拟”的方式扩展维度
。
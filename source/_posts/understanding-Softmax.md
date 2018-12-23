title: understanding Softmax in CNN
date: 2018-11-18 21:24:51
tags:
math: true
---

Softmax层在CNN网络结构中经常放在全连接层后面，做为分类器，输出一个概率值向量。
它主要用来进行多分类，形式简单，用一个公式$\frac{e^{x}}{\sum_{i=1}^{n}e^{x}}$计算，但是为什么
是这样的形式呢？

## 直观的理解方式

首先，Softmax层的作用就是，将一个特征向量变成一个概率值向量。比如一幅图片，在
经过卷积层以 后，得到一个特征向量，这个特征向量可以认为是输入的一种表示方法，
然后经过Softmax层，对应到输出概率，这个输出概率的大小能反映出输入属于 哪种分类
的概率最大。

这种将一个向量转换成概率向量的变换需要满足一些条件，

1. **输出概率值的和应该等于1**，那么我们可以定义
   $Softmax(X)=g(X)=\frac{f(x)}{\Sigma{f(x)}}$。
2. **$g(x)$ 是单调的**。变换 $g(x)$ 要能准确的反映输入 $x$ 的相对大小，比如，
   输入 $x$ 分别是 [1, 2, 3] , 那么输出 $g(x)$ 可以是 [1/6, 2/6, 3/6]。
3. **函数的输出域是(0,$\infty$)**。考虑直接取 $f(x)=x$，则如果输入当中有负数，
   概率值就会是负的，显然不合常理。所以需要让每一个概率值都大于0。

反之，取 $f(x)=e^x$ 能满足上述所有要求，函数是单调的且输出域大于0，从而
$Softmax(x)=\frac{e^{x}}{\sum_{i=1}^{n}e^{x}}$。但是符合单调函数且输出域大于0
的函数那么多， 为什么没有选用其他的满足条件的函数呢？

其实，选用$Softmax(x)=\frac{e^{x}}{\sum_{i=1}^{n}e^{x}}$ 是必要的，这样的形式
是有理论基础和相关证明的，下面就挑战一下，去理解 Softmax 背后的统计学原理吧。

## Softmax 的统计学证明
在进行证明前，首先我们需要理解广义线性模型。我们知道线性模型就是$y=W^TX$的形
式，代表了对输入向量的线性变换，而如果用 $y=g(W^TX)$ 这样的函数，把线性模型的
结果作为参 数，且$g$属于指数分布族，那就是广义线性的。广义线性模型的具体定义如
下：

### 广义线性模型的具体定义
1. 给定 $x$ 和参数 $\theta$, $y|x$ 满足以 $\eta$ 为变量的指数分布族；
2. 参数 $\eta=\theta^Tx$；
3. 给定 $x$, 我们需要预测 $T(y)$, 通常 $T(y)=y$。

### 指数分布族的定义
指数分布族是一类概率密度函数可以写成 $P(y;\eta) = b(y) exp(\eta^TT(y)-a(\eta))$
形式的分布的总称，正态分布、伯努利分布、多项式分布都属于指数分布族。

### 从指数分布族推导对应的广义线性模型
1. $Y|X;\theta \sim expfamily(\eta)$，假设给定 $X$ 和参数 $\theta$，对应的参数 $Y$ 服从一个以 $\eta$ 为参数的指数分布族分布。
2. 给定 $X$，目标为 $T(Y)$，通常 $T(Y)=Y$。
3. $\eta=\theta^TX$，假设指数分布族的参数是 $X$ 的线性加和。
4. 根据指数分布族的概率密度函数 $P(y;\eta)=b(y)exp(\eta^TT(y)-a(\eta))$，推导 $\eta$、$b(y)$、$T(y)$、$a(\eta)$。
5. 根据 $\eta$ 和指数分布族的期望推导 $Y$ 的表达式。

### Softmax的推导

1. 假设特征到概率服从一个多项式分布，则概率密度函数是
$P(Y|X;\phi_1,\phi_2,\dots,\phi_k)=\phi_1^{\Delta(Y=1)} \ast
\phi_2^{\Delta(Y=2)} \ast \dots \ast \phi_k^{\Delta(Y=k)}$ ， 其中
$\Delta(Y=i)$ 表示输出 $Y$ 值是否等于第 $i$ 个分类，若是，则函数值为 $1$
，若不是，则函数值为 $0$ 。

2. 用指数分布族的形式代，且$Y(i)=\Delta(Y=i)$，则可以推导出：
$$
\begin{align}
P(Y|X;\phi_1,\phi_2,\dots,\phi_k) &= \phi_1^{T(1)} \ast \phi_2^{T(2)} \ast \dots \ast \phi_k^{T(k)} \\\\
&=exp\left(T(1) \ast log(\phi_1)+T(2) \ast log(\phi_2)+ \dots +(1-\sum_i^{k-1} T(i)) \ast log(\phi_k)\right) \\\\
&=exp\left( T(1) \ast log(\frac{\phi_1}{\phi_k})+T(2) \ast log(\frac{\phi_2}{\phi_k})+ \dots +log(\phi_k) \right) \\\\
&=b(y) exp\left( \eta^TT(y)-a(\eta) \right)
\end{align}
$$

3. 得出$b(y)=1$，$\eta=\left[ log(\frac{\phi_1}{\phi_k}),log(\frac{\phi_2}{\phi_k}),\dots,log(\frac{\phi_{k-1}}{\phi_k}) \right]$，$a(\eta)=-log(\phi_k)$。

4. 这个分布的期望是 $E(Y_i|X)=\phi_i$，而 $\eta_i=log(\frac{\phi_i}{\phi_k})$ ，那么可以推出

$$
e^\eta_i = \frac{\phi_i}{\phi_k} \\\\
\phi_k \ast e^\eta_i = \phi_i \\\\
\phi_k \ast \sum_{i=1}^{k}e^\eta_i = \sum_{i=1}^{k}{\phi_i} = 1 \\\\
\therefore \phi_i = \frac{e^\eta_i}{\sum_{j=1}^{k}{e^\eta_j}}
$$

5. 广义线性模型的假设第三条 $\eta=\theta^TX$ ，推导出期望 $E(Y_i|X) = \phi_i =
\frac{ e^{\theta_i^TX} }{ \sum_{j=1}^{k}{e^{\theta_j^TX} } }$
，就是我们 Softmax 的输出啦！

### Reference
[1] https://www.cnblogs.com/yinheyi/p/6131262.html

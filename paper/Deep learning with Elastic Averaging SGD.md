https://arxiv.org/pdf/1412.6651.pdf

## Abstract
We study the problem of stochastic optimization for deep learning in the parallel computing environment under communication constraints. A new algorithm is proposed in this setting where the communication and coordination of work among concurrent processes (local workers), is based on an elastic force which links the parameters they compute with a center variable stored by the parameter server (master). The algorithm enables the local workers to perform more exploration, i.e. the algorithm allows the local variables to fluctuate further from the center variable by reducing the amount of communication between local workers and the master. We empirically demonstrate that in the deep learning setting, due to the existence of many local optima, allowing more exploration can lead to the improved performance. We propose synchronous and asynchronous variants of the new algorithm. We provide the stability analysis of the asynchronous variant in the round-robin scheme and compare it with the more common parallelized method ADMM. We show that the stability of EASGD is guaranteed when a simple stability condition is satisfied, which is not the case for ADMM. We additionally propose the momentum-based version of our algorithm that can be applied in both synchronous and asynchronous settings. Asynchronous variant of the algorithm is applied to train convolutional neural networks for image classification on the CIFAR and ImageNet datasets. Experiments demonstrate that the new algorithm accelerates the training of deep architectures compared to DOWNPOUR and other common baseline approaches and furthermore is very communication efficient.

 我们研究了通信约束下并行计算环境中深度学习的随机优化问题。在该设置中提出了一种新算法，其中并发过程（本地工作者）之间的工作的通信和协调基于弹性力，该弹性力将它们计算的参数与由参数服务器（主）存储的中心变量链接。该算法使本地工作人员能够进行更多探索，即该算法通过减少本地工作人员与主人之间的通信量，允许本地变量从中心变量进一步波动。我们凭经验证明，在深度学习环境中，由于存在许多局部最优，允许更多的探索可以导致改进的性能。我们提出了新算法的同步和异步变体。我们在循环方案中提供异步变量的稳定性分析，并将其与更常见的并行方法ADMM进行比较。我们表明，当满足简单的稳定条件时，EASGD的稳定性得到保证，而ADMM则不然。我们还提出了基于动量的算法版本，可以在同步和异步设置中应用。该算法的异步变体应用于训练卷积神经网络，用于CIFAR和ImageNet数据集上的图像分类。实验表明，与DOWNPOUR和其他常见的基线方法相比，新算法加速了深层架构的训练，并且还具有非常高的通信效率。
 ## Introduction
One of the most challenging problems in large-scale machine learning is how to parallelize the training of large models that use a form of stochastic gradient descent (SGD) [1]. There have been attempts to parallelize SGD-based training for large-scale deep learning models on large number of CPUs, including the Google’s Distbelief system [2]. But practical image recognition systems consist of large-scale convolutional neural networks trained on few GPU cards sitting in a single computer [3, 4]. The main challenge is to devise parallel SGD algorithms to train large-scale deep learning models that yield a significant speedup when run on multiple GPU cards.

 大规模机器学习中最具挑战性的问题之一是如何并行化使用随机梯度下降（SGD）形式的大型模型的训练[1]。已经尝试在大量CPU上并行化基于SGD的大规模深度学习模型训练，包括谷歌的Distbelief系统[2]。但实际的图像识别系统包括大规模的卷积神经网络，这些神经网络只需要坐在一台计算机上的少量GPU卡上进行训练[3,4]。主要的挑战是设计并行的SGD算法来训练大规模深度学习模型，这些模型在多个GPU卡上运行时可以产生显着的加速。
 
 In this paper we introduce the Elastic Averaging SGD method (EASGD) and its variants. EASGD is motivated by quadratic penalty method [5], but is re-interpreted as a parallelized extension of the averaging SGD algorithm [6]. The basic idea is to let each worker maintain its own local parameter, and the communication and coordination of work among the local workers is based on an elastic force which links the parameters they compute with a center variable stored by the master. The center variable is updated as a moving average where the average is taken in time and also in space over the parameters computed by local workers. 
 
 在本文中，我们介绍了弹性平均SGD方法（EASGD）及其变体。 EASGD受二次惩罚法[5]的推动，但被重新解释为平均SGD算法的并行扩展[6]。基本思想是让每个工人保持自己的本地参数，并且本地工人之间的工作的沟通和协调基于弹性力，该弹性力将他们计算的参数与由主人存储的中心变量相关联。中心变量更新为移动平均值，其中平均值在时间上以及在本地工作人员计算的参数的空间中。
 
 The main contribution of this paper is a new algorithm that provides fast convergent minimization while outperforming DOWNPOUR method [2] and other baseline approaches in practice. Simultaneously it reduces the communication overhead between the master and the local workers while at the same time it maintains high-quality performance measured by the test error. The new algorithm applies to deep learning settings such as parallelized training of convolutional neural networks.
 
 本文的主要贡献是提供快速收敛最小化的新算法，同时在实践中优于DOWNPOUR方法[2]和其他基线方法。同时，它减少了主设备和本地工作人员之间的通信开销，同时保持了由测试错误测量的高质量性能。新算法适用于深度学习设置，如卷积神经网络的并行训练。
 
 The article is organized as follows. Section 2 explains the problem setting, Section 3 presents the synchronous EASGD algorithm and its asynchronous and momentum-based variants, Section 4 provides stability analysis of EASGD and ADMM in the round-robin scheme, Section 5 shows experimental results and Section 6 concludes. The Supplement contains additional material including additional theoretical analysis.
 
 文章的结构安排如下。第2节解释了问题设置，第3节介绍了同步EASGD算法及其异步和基于动量的变量，第4节提供了循环方案中EASGD和ADMM的稳定性分析，第5节显示了实验结果，第6节得出结论。补充材料包含其他材料，包括额外的理论分析。
 ## Problem setting
The problem of the equivalence of these two objectives is studied in the literature and is known as the augmentability or the global variable consensus problem.The quadratic penalty term ρ in Equation 2 is expected to ensure that local workers will not fall into different attractors that are far away from the center variable. This paper focuses on the problem of reducing the parameter communication overhead between the master and local workers [10, 2, 11, 12, 13]. The problem of data communication when the data is distributed among the workers [7, 14] is a more general problem and is not addressed in this work. We however emphasize that our problem setting is still highly non-trivial under the communication constraints due to the existence of many local optima [15].

 这里还有些内容，数学公式。
这两个目标的等价问题在文献中被研究并被称为可扩充性或全局变量共识问题。方程2中的二次惩罚项ρ有望确保本地工人不会陷入不同的吸引子。 远离中心变量。 本文重点讨论减少主工人和本地工人之间参数通信开销的问题[10,2,11,12,13]。 当数据在工作人员之间分配时数据通信的问题[7,14]是一个更普遍的问题，在这项工作中没有解决。 然而，我们强调，由于存在许多局部最优[15]，在通信约束下我们的问题设置仍然是非常重要的。
 ## EASGD update rule

特别是，小ρ允许更多的探索，因为它允许xi从中心x~进一步波动。 EASGD的独特理念是允许当地工人进行更多的勘探（小ρ）和主人进行开采。 这种方法不同于文献[2,17,18,19,20,21,22,23]中探讨的其他设置，并关注中心变量收敛的速度。 在本文中，我们展示了我们在深度学习环境中的优点。

### Asynchronous EASGD
我们在前一节讨论了EASGD算法的同步更新。 在本节中，我们提出了它的异步变体。 本地工作人员仍然负责更新局部变量xi，而主人员正在更新中心变量x~。每个工作者都保持自己的时钟ti，从0开始，并在每次随机梯度更新xi后递增1，如算法1所示。无论何时本地工作人员完成梯度更新的τ步骤，主人都执行更新，我们参考 以τ作为沟通期。 从算法1中可以看出，每当τ除以第i个工作者的本地时钟时，第i个工作者与主服务器通信并请求中心变量x的当前值.

然后工人等待，直到master发回所请求的参数值，并计算弹性差α（x-x~）（在算法1的步骤a中捕获整个过程）。 然后将弹性差发送回算法1中的master（步骤b），然后更新x~。通信周期τ控制每个本地工作人员和主人之间的通信频率，从而控制探索和利用之间的权衡。

### Momentum EASGD
动量EASGD（EAMSGD）是我们的算法1的变体，并在算法2中捕获。它基于Nesterov的动量方案[24,25,26]，其中更新了公式3中捕获的形式的本地工作者 由以下更新替换:数学公式。其中δ是动量项。 注意，当δ= 0时，我们恢复原始的EASGD算法。

由于我们有兴趣减少参数向量非常大的并行计算环境中的通信开销，我们将在实验部分探索异步EASGD算法及其在相对较大的τ机制中的基于动量的变量（通信频率较低）。

## Stability analysis of EASGD and ADMM in the round-robin scheme
在本节中，我们研究了循环方案中异步EASGD和ADMM方法的稳定性[20]。 我们首先在此设置中说明两种算法的更新，然后我们研究它们的稳定性。 我们将证明在一维二次情形中，ADMM算法可以表现出混沌行为，导致指数发散。 ADMM算法稳定的分析条件仍然未知，而对于EASGD算法则非常简单。

在二次和强凸的情况下，同步EASGD算法的分析，包括其收敛速度和平均属性，推迟到补充。

在我们的设置中，ADMM方法[9,27,28]涉及解决以下最小极大问题2。

下面大量数学公式GG

## 实验
在本节中，我们将EASGD和EAMSGD的性能与并行方法DOWNPOUR和顺序方法SGD以及它们的平均和动量变量进行比较。

下面列出了所有并行比较器方法:

DOWNPOUR，本文中使用的DOWNPOUR实现的伪代码包含在Supplement中。

动量下降（MDOWNPOUR），其中Nesterov的动量方案应用于主人的更新（注意不清楚如何将其应用于当地工人或τ> 1的情况）。 伪代码在补充中。

### Experimental setup
对于我们的所有实验，我们使用与InfiniBand互连的GPU集群。每个节点都有4个Titan GPU处理器，每个本地工作者对应一个GPU处理器。主集中心变量在集中参数服务器上存储和更新。

为了描述卷积神经网络的体系结构，我们将首先介绍一种表示法。设（c，y）表示每层输入图像的大小，其中c是颜色通道的数量，y是输入的水平和垂直尺寸。Let C denotes the fully-connected convolutional operator and let P denotes the max pooling operator, D denotes the linear operator with dropout rate equal to 0.5 and S denotes the linear operator with softmax output non-linearity.(设C表示完全连接的卷积算子，让P表示最大合并算子，D表示线性算子，其丢失率等于0.5，S表示具有softmax输出非线性的线性算子。)We use the cross-entropy loss and all inner layers use rectified linear units. (我们使用交叉熵损失，所有内层使用整流线性单元。)

对于ImageNet实验，我们使用与[4]相似的方法，使用以下11层卷积神经网络（3,221）C（96,108）P（96,36）C（256,32）P（256,16）C（384），14）
C（384,13）C（256,12）P（256,6）d（4096,1）d（4096,1）S（1000,1）。 对于CIFAR实验，我们使用与[29]相似的方法，使用以下7层卷积神经网络（3,28）C（64,24）P（64,12）C（128,8）P（128,4））C（64,2）d（256,1）S（10,1）。

在我们的实验中，我们运行的所有方法都使用随机选择的相同初始参数，除了我们将CIFAR情况下的所有偏差设置为零，而ImageNet情况设置为0.1。 此参数用于初始化 the master and all the local workers.

我们将l2-正则化加到损失函数F（x）。 对于ImageNet，我们使用λ= 10-5.对于CIFAR，我们使用λ= 10-4。 我们还使用样本大小为128的小批量计算随机梯度。

### Experimental results
对于本节中的所有实验，我们使用β= 0.9的EASGD，对于所有基于动量的方法，我们设置动量项δ= 0.99，最后对于MVADOWNPOUR，我们将移动速率设置为α= 0.001。我们从CIFAR数据集上的实验开始，在单个计算节点上运行p = 4个本地工作程序。对于所有方法，我们从以下集合τ= {1,4,16,64}检查了通信周期。为了进行比较，我们还报告了MSGD的性能，其表现优于SGD，ASGD和MVASGD，如补充中的图6所示。对于每种方法，我们检查了广泛的学习率（所有实验中探索的学习率总结在补编中的表1,2,3中）。CIFAR实验独立于相同的初始化运行3次，并且对于每种方法，我们报告其通过可实现的最小测试误差测量的最佳性能。从图2中的结果可以得出结论，所有基于DOWNPOUR的方法对于小τ（τ∈{1,4}）都达到了最佳性能（测试误差），并且对于τ∈{16,64}变得高度不稳定。虽然通过更快的收敛，EAMSGD明显优于所有τ值的比较器方法。它还可以找到由测试误差测量的更好质量的解决方案，这种优势对于τ∈{16,64}变得更加重要。注意，用更大的τ实现更好的测试性能的趋势也是EASGD算法的特征。

接下来，我们将从集合p = {4,8,16}中为CIFAR实验探索不同数量的本地工作人员p，并且对于ImageNet实验p = {4,8}。 对于ImageNet实验，我们使用我们找到的最佳设置报告一次运行的结果。 EASGD和EAMSGD以τ= 10运行，而DOWNPOUR和MDOWNPOUR以τ= 1运行。结果如图3和图4所示。对于CIFAR实验，EASGD或EAMSGD可达到的最低测试误差随着较大而降低页。 这可以通过以下事实来解释：较大的p允许更多地探索参数空间。 在补编中，我们进一步讨论了探索和利用之间的权衡取决于学习率（第9.5节）和通信期（第9.6节）。 最后，ImageNet实验的结果也显示了EAMSGD优于竞争对手方法的优势。

## Conclusion
在本文中，我们描述了一种称为EASGD的新算法及其用于在随机设置中训练深度神经网络的变体，当计算在多个GPU上并行化时。 实验表明，与更常见的基线方法（如DOWNPOUR及其变体）相比，这种新算法可以快速实现测试误差的改善。 我们表明，我们的方法在通信限制下非常稳定和合理。 我们在循环方案中提供了异步EASGD的稳定性分析，并展示了该方法相对于ADMM的理论优势。 EASGD算法与其基于矩阵的变量EAMSGD的不同行为是有趣的，并将在未来的工作中进行研究。


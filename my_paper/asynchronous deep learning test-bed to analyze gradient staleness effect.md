## Asynchronous Deep Learning Test-bed to Analyze Gradient Staleness Effect 异步深度学习试验台分析梯度过时效应
### Abstract
For modern machine learning systems, including deep learning systems, parallelization is inevitable since they are required to process massive amount of training data. One of the hot topic of this area is the data parallel learning where multiple nodes cooperate each other exchanging parameter / gradient periodically. In order to efficiently implement data-parallel machine learning in a collection of computers with a relatively sparse network, it is indispensable to asynchronously update model parameters through gradients, but the effect of the learning model through asynchronous analysis has not yet been fully understood. In this paper, we propose a software test-bed for analyzing gradient staleness effect on prediction performance, using deep learning framework TensorFlow and distributed computing framework Ray. We report the architecture of the test-bed and initial evaluation results.

对于现代机器学习系统，包括深度学习系统，并行化是不可避免的，因为它们需要处理大量的训练数据。 该领域的热门话题之一是数据并行学习，其中多个节点彼此协作，周期性地交换参数/梯度。 为了在具有相对稀疏网络的计算机集合中有效地实现数据并行机器学习，通过梯度异步更新模型参数是必不可少的，但是学习模型通过异步分析的效果尚未完全理解。 在本文中，我们提出了一个软件测试平台，用于分析梯度过期对预测性能的影响，使用深度学习框架TensorFlow和分布式计算框架Ray。 我们报告了试验台的结构和初步评估结果。
### 1,Introduction
For modern machine learning systems, including deep learning systems, parallelization is inevitable since they are required to process massive amount of training data.

Machine learning systems that use the data parallel method train multiple machine learning models simultaneously with different training data subset, and synchronize the machine learning models periodically. There are mainly two methods to achieve the synchronization of models, i.e., the exchange of parameters which constitute the models, or gradients of parameters. One method uses central server called parameter server to synchronize the parameter, and the other method lets the worker nodes directly communicate and exchange information with each other to synchronize as whole.

对于现代机器学习系统，包括深度学习系统，并行化是不可避免的，因为它们需要处理大量的训练数据。

使用数据并行方法的机器学习系统同时训练多个机器学习模型和不同的训练数据子集，并定期同步机​​器学习模型。主要有两种方法来实现模型的同步，即构成模型的参数的交换，或参数的梯度。一种方法使用称为参数服务器的中央服务器来同步参数，另一种方法允许工作者节点直接相互通信和交换信息以进行整体同步。

In large-scale data parallel machine learning, since the number of nodes increases, the batch size generally becomes large, and the gradient staleness becomes a problem. Furthermore, this problem becomes more prominent in asynchronous parallel machine learning which does not synchronize for each batch. In an efficient data parallel machine learning system, it is indispensable to adjust the learning algorithm and hyper-parameter to mitigate the gradient staleness, but these relationships are not generally understood.

In order to understand the relationship between gradient staleness, learning algorithm and hyper-parameter, we are developing a kind of simulation environment which can reproduce propagation delay of gradient.

在大规模数据并行机器学习中，由于节点数量增加，批量大小通常变大，并且梯度过时成为问题。此外，这个问题在异步并行机器学习中变得更加突出，它不会为每个批次同步。在高效的数据并行机器学习系统中，调整学习算法和超参数以减轻梯度过时是必不可少的，但这些关系通常不被理解。

为了理解梯度过时，学习算法和超参数之间的关系，我们正在开发一种能够再现梯度传播延迟的仿真环境。


The contribution of the paper are the follows. We describe the parallel machine learning environment constructed as the prototype of the simulation environment construction. In this simulation environment, it is possible to arbitrarily set the update delay of the gradient in asynchronous parallel machine learning. As a result of experiments with several update delays, we confirmed the influence on the machine learning process due to the expansion of delay.

该论文的贡献如下。 我们将并行机器学习环境描述为模拟环境构造的原型。 在该仿真环境中，可以在异步并行机器学习中任意设置梯度的更新延迟。 由于多次更新延迟的实验，我们确认了由于延迟的扩大对机器学习过程的影响。

The next section of this paper gives the overview of distributed machine learning systems focusing on the parameter exchange methods, the introduction of stochastic gradient descent and the distributed execution framework Ray that we use in this paper to allow distributed applications to run the same code on a single machine for efficient multi-processing.Next section presents how we implement the synchronous and asynchronous parameter servers using Ray, and how to design the test-bed for analyzing gradient staleness effect.

本文的下一部分概述了分布式机器学习系统，重点介绍了参数交换方法，随机梯度下降的引入以及我们在本文中使用的分布式执行框架Ray，以允许分布式应用程序在 用于高效多处理的单机。下一节介绍如何使用Ray实现同步和异步参数服务器，以及如何设计用于分析梯度过时效果的测试台。
### 2. Background
#### 2. 1 Parameter Exchange Methods for Large Scale Machine Learning Systems
To parallelize machine learning systems, there are two methods;Data Parallel and Model Parallel.
While data parallel method simultaneously trains multiple machine learning models synchronizing each other, model parallel parallelize inside a single machine learning model. While these two methods are not exclusive each other and often used complementarily, we focus on data parallel in this paper.Data parallel machine learning methods could be categorized into two types; synchronous methods and asynchronous methods; synchronous means all the machine learning models are strictly becomes same periodically, while asynchronous methods allow slight difference among the models.To implement data parallel machine learning systems, there are two methods: parameter server based method and direct communication method.

为了并行化机器学习系统，有两种方法：数据并行和模型并行。
虽然数据并行方法同时训练多个机器学习模型彼此同步，但模型在单个机器学习模型内并行化。 虽然这两种方法并不是互相排斥的，而是经常互补使用，但我们关注的是本文中并行数据。数据并行机器学习方法可以分为两类： 同步方法和异步方法; 同步意味着所有的机器学习模型都是周期性地严格相同，而异步方法允许模型之间略有不同。为了实现数据并行机器学习系统，有两种方法：基于参数服务器的方法和直接通信方法。
#### 2. 1. 1 Parameter Server based Method
Central server to exchange parameters are often called parameter server. The left diagram in Figure 1 shows the parameter server based parameter exchange. The workers (machine learning modules) send parameters (or gradients) to the parameter server, the parameter server aggregates the parameter, and send back them to the workers. Often, multiple parameter servers are used to shrad the parameters and balance the load; each parameter server take care of a certain subset of parameters.

中央服务器交换参数通常称为参数服务器。 图1中的左图显示了基于参数服务器的参数交换。 工作人员（机器学习模块）将参数（或梯度）发送到参数服务器，参数服务器聚合参数，并将它们发送回工作人员。 通常，多个参数服务器用于调整参数并平衡负载; 每个参数服务器负责某个参数子集。
#### 2. 1. 2 Direct Exchange Method
It is possible to synchronize the models without using central server. By repeating peer-to-peer exchange of parameters.

可以在不使用中央服务器的情况下同步模型。 通过重复参数的对等交换。

#### 2.2 Stochastic Gradient Descent(SGD)
Both statistical estimation and machine learning consider the problem of minimizing an objective function with summation form:统计估计和机器学习都考虑了用求和形式最小化目标函数的问题：

where the parameter theta which minimizes J(θ) is to be estimated. Each sum function Ji is typically associated with the i-th observation value(for training) in the dataset.

In the machine learning algorithm, when the loss function is minimized, it can be iteratively solved step by step through the gradient descent method to obtain a minimized loss function and model parameter values. Gradient Descent Optimization is the most commonly used optimization algorithm for neural network model training. For the deep learning model, the gradient descent algorithm is basically used for optimization training.

The principle of the gradient descent algorithm: The gradient of the objective function J(θ) with respect to the parameter θ will be the fastest rising direction of the objective function. For the minimization optimization problem, it is only necessary to advance the parameters one step in the opposite direction of the gradient, and then the objective function can be reduced. This step size is also known as learning rate η.

其中，要估计最小化J（θ）的参数θ。每个和函数Ji通常与数据集中的第i个观察值（用于训练）相关联。

在机器学习算法中，当损失函数最小化时，可以通过梯度下降法逐步迭代地求解，以获得最小化的损失函数和模型参数值。梯度下降优化是神经网络模型训练中最常用的优化算法。对于深度学习模型，梯度下降算法基本上用于优化训练。

梯度下降算法的原理：目标函数J（θ）相对于参数θ的梯度将是目标函数的最快上升方向。对于最小化优化问题，仅需要在梯度的相反方向上将参数前进一步，然后可以减小目标函数。该步长也称为学习率η。

When used to minimize the above function, a standard (or "batch") gradient descent method would perform the following iterations :

Where ∇J(θ) is the gradient of the parameter, according to the difference in the amount of data used to calculate the objective function J(θ), the gradient descent algorithm can be divided into Batch Gradient Descent, Stochastic Gradient Descent and Mini-batch Gradient Descent. For the batch gradient descent algorithm, the J(θ) is calculated over the entire training set. If the data set is large, it may face the problem of insufficient memory, and its convergence speed is generally slow. The stochastic gradient descent algorithm is another case. J(θ) is calculated for a training sample in a training set. That is, a sample is obtained and a parameter update can be performed. 

当用于最小化上述函数时，标准（或“批处理”）梯度下降方法将执行以下迭代：

其中∇J（θ）是参数的梯度，根据用于计算目标函数J（θ）的数据量的差异，梯度下降算法可分为批量梯度下降，随机梯度下降和迷你-batch Gradient Descent。对于批量梯度下降算法，在整个训练集上计算J（θ）。如果数据集很大，则可能面临内存不足的问题，并且其收敛速度通常较慢。随机梯度下降算法是另一种情况。针对训练集中的训练样本计算J（θ）。也就是说，获得样本并且可以执行参数更新。

Therefore, the convergence speed will be faster, but there may be fluctuations in the value of the objective function because high-frequency parameter updates result in high variance. The Mini-batch gradient descent algorithm is a compromise solution. Selecting a small batch of samples in the training set to calculate J(θ) can ensure that the training process is more stable, and that the batch training method can also use the advantage of matrix calculations. This is the most commonly used gradient descent algorithm. We focus on stochastic gradient descent in this paper.

因此，收敛速度将更快，但是目标函数的值可能存在波动，因为高频参数更新导致高方差。 Mini-batch梯度下降算法是折衷解决方案。在训练集中选择一小批样本来计算J（θ）可以确保训练过程更加稳定，并且批量训练方法也可以利用矩阵计算的优势。这是最常用的梯度下降算法。本文着重研究随机梯度下降。

In stochastic gradient descent, the true gradient of J(θ) is approximated by a gradient at a single example:在随机梯度下降中，J（θ）的真实梯度在一个例子中用梯度近似：

As the algorithm sweeps through the training set, it will perform above update for each training example. It can be done several times on the training set until the algorithm converges. If this is done, each transmission data can be shuffled to prevent cycles.

当算法扫过训练集时，它将针对每个训练示例执行上述更新。它可以在训练集上多次完成，直到算法收敛。如果这样做，则可以对每个传输数据进行混洗以防止循环。

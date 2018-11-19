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
Both statistical estimation and machine learning consider the problem of minimizing an objective function with summation form:统计估计和机器学习都考虑了用求和形式最小化目标函数的问题： **公式1**

where the parameter theta which minimizes J(θ) is to be estimated. Each sum function Ji is typically associated with the i-th observation value(for training) in the dataset.

In the machine learning algorithm, when the loss function is minimized, it can be iteratively solved step by step through the gradient descent method to obtain a minimized loss function and model parameter values. Gradient Descent Optimization is the most commonly used optimization algorithm for neural network model training. For the deep learning model, the gradient descent algorithm is basically used for optimization training.

The principle of the gradient descent algorithm: The gradient of the objective function J(θ) with respect to the parameter θ will be the fastest rising direction of the objective function. For the minimization optimization problem, it is only necessary to advance the parameters one step in the opposite direction of the gradient, and then the objective function can be reduced. This step size is also known as learning rate η.

其中，要估计最小化J（θ）的参数θ。每个和函数Ji通常与数据集中的第i个观察值（用于训练）相关联。

在机器学习算法中，当损失函数最小化时，可以通过梯度下降法逐步迭代地求解，以获得最小化的损失函数和模型参数值。梯度下降优化是神经网络模型训练中最常用的优化算法。对于深度学习模型，梯度下降算法基本上用于优化训练。

梯度下降算法的原理：目标函数J（θ）相对于参数θ的梯度将是目标函数的最快上升方向。对于最小化优化问题，仅需要在梯度的相反方向上将参数前进一步，然后可以减小目标函数。该步长也称为学习率η。

When used to minimize the above function, a standard (or "batch") gradient descent method would perform the following iterations: **公式2**

Where ∇J(θ) is the gradient of the parameter, according to the difference in the amount of data used to calculate the objective function J(θ), the gradient descent algorithm can be divided into Batch Gradient Descent, Stochastic Gradient Descent and Mini-batch Gradient Descent. For the batch gradient descent algorithm, the J(θ) is calculated over the entire training set. If the data set is large, it may face the problem of insufficient memory, and its convergence speed is generally slow. The stochastic gradient descent algorithm is another case. J(θ) is calculated for a training sample in a training set. That is, a sample is obtained and a parameter update can be performed. 

当用于最小化上述函数时，标准（或“批处理”）梯度下降方法将执行以下迭代：

其中∇J（θ）是参数的梯度，根据用于计算目标函数J（θ）的数据量的差异，梯度下降算法可分为批量梯度下降，随机梯度下降和迷你-batch Gradient Descent。对于批量梯度下降算法，在整个训练集上计算J（θ）。如果数据集很大，则可能面临内存不足的问题，并且其收敛速度通常较慢。随机梯度下降算法是另一种情况。针对训练集中的训练样本计算J（θ）。也就是说，获得样本并且可以执行参数更新。

Therefore, the convergence speed will be faster, but there may be fluctuations in the value of the objective function because high-frequency parameter updates result in high variance. The Mini-batch gradient descent algorithm is a compromise solution. Selecting a small batch of samples in the training set to calculate J(θ) can ensure that the training process is more stable, and that the batch training method can also use the advantage of matrix calculations. This is the most commonly used gradient descent algorithm. We focus on stochastic gradient descent in this paper.

因此，收敛速度将更快，但是目标函数的值可能存在波动，因为高频参数更新导致高方差。 Mini-batch梯度下降算法是折衷解决方案。在训练集中选择一小批样本来计算J（θ）可以确保训练过程更加稳定，并且批量训练方法也可以利用矩阵计算的优势。这是最常用的梯度下降算法。本文着重研究随机梯度下降。

In stochastic gradient descent, the true gradient of J(θ) is approximated by a gradient at a single example:在随机梯度下降中，J（θ）的真实梯度在一个例子中用梯度近似： **公式3**

As the algorithm sweeps through the training set, it will perform above update for each training example. It can be done several times on the training set until the algorithm converges. If this is done, each transmission data can be shuffled to prevent cycles.

当算法扫过训练集时，它将针对每个训练示例执行上述更新。它可以在训练集上多次完成，直到算法收敛。如果这样做，则可以对每个传输数据进行混洗以防止循环。
#### 2. 2. 1 Synchronous Stochastic Gradient Descent(SSGD)
As we mentioned in the 2.1 section, data parallel machine learning methods could be categorized into two types; synchronous methods and asynchronous methods. So, for synchronous SGD means that each computer used for parallel computing calculates the gradient value after calculating its own batch, and sends the gradient value to parameter server. The parameter server obtains the gradient average and updates the parameters on the parameter server. 

As shown in Figure 2, it can be seen as four computers. The first computer is used to store parameters, share parameters, and share calculations. It can be simply understood as a memory and computing shared area, that is the parameter server job; The other three computers are used for parallel computing to calculate the gradient value, which is worker task.

正如我们在2.1节中提到的，数据并行机器学习方法可以分为两类： 同步方法和异步方法。 因此，对于同步SGD，意味着用于并行计算的每台计算机在计算自己的批次后计算梯度值，并将梯度值发送到参数服务器。 参数服务器获取梯度平均值并更新参数服务器上的参数。

如图2所示，它可以看作是四台计算机。 第一台计算机用于存储参数，共享参数和共享计算。 它可以简单地理解为内存和计算共享区域，即参数服务器作业; 其他三台计算机用于并行计算以计算梯度值，这是工作任务。

The disadvantage of this method of calculation is that each gradient update must be waited until all workers A, B, and C have been calculated before updating the parameters. That is, the speed of iterative update depends on the slowest worker among the three A, B, and C workers. Therefore, the method of simultaneous update is recommended to have the same computing power.

这种计算方法的缺点是必须等待每个梯度更新，直到在更新参数之前计算了所有工人A，B和C. 也就是说，迭代更新的速度取决于三个A，B和C工人中最慢的工人。 因此，建议同时更新的方法具有相同的计算能力。
#### 2. 2. 2 Asynchronous Stochastic Gradient Descent(ASGD)
The parameter server receives the gradient value of a machine as soon as it receives the parameter update, without waiting for other machines. This iterative method is relatively unstable, and the convergence curve is more severe, because when the worker A updates the parameters in the parameter server, it may be that the worker B is still using the old parameter values of the previous iteration.

一旦接收到参数更新，参数服务器就会立即接收机器的梯度值，而无需等待其他机器。 该迭代方法相对不稳定，并且收敛曲线更严重，因为当工作者A更新参数服务器中的参数时，工作者B可能仍在使用前一次迭代的旧参数值。
#### 2. 3 Ray: a fiexible, high-performance distributed execution framework 一个灵活，高性能的分布式执行框架
Ray is a Python-based distributed execution engine designed for large-scale machine learning and reinforcement learning applications. It is fully compatible with deep learning frameworks like TensorFlow, PyTorch. The goal of ray is to allow high-performance distributed applications to run the same code on a single machine for efficient multi-processing and it can be used on a cluster for large computations.And also, it can enable machine learning and deep learning workloads to be executed in real time with MPI-like power and granularity. It uses shared-memory distributed object storage to process big data efficiently and uses a bottom-up hierarchical scheduling architecture to implement low-latency and high-throughput scheduling.

Ray is fast, micro-second latencies for a single task, and can handle heterogeneous hardware, with some application workloads being executed on CPUs and others executing on GPUs. It has many schedulers that can group all of these together. It also borrows the task dependency properties from MPI. Ray will maintain the computational state among the nodes in the cluster, but with as few states as possible, which will maximize robustness.

Ray是一个基于Python的分布式执行引擎，专为大型机器学习和强化学习应用程序而设计。它与TensorFlow，PyTorch等深度学习框架完全兼容。 ray的目标是允许高性能分布式应用程序在单个机器上运行相同的代码以实现高效的多处理，并且可以在群集上用于大型计算。此外，它还可以实现机器学习和深度学习工作负载以类似MPI的功能和粒度实时执行。它使用共享内存分布式对象存储来高效处理大数据，并使用自下而上的分层调度架构来实现低延迟和高吞吐量调度。

Ray是单个任务的快速，微秒级延迟，可以处理异构硬件，其中一些应用程序工作负载在CPU上执行，而其他应用程序在GPU上执行。它有许多调度程序可以将所有这些组合在一起。它还借用了MPI的任务依赖属性。 Ray将维持集群中节点之间的计算状态，但是具有尽可能少的状态，这将最大化鲁棒性。
### 3. Methodology 
We implement synchronous and asynchronous parameter servers using distributed actor handles, which are still considered experimental. For actors, Ray uses actors to extend the dataflow model. An actor is essentially a stateful worker(or a service). When a new actor is instantiated, a new worker is created, and the actor’s methods are arranged on that specific worker and can access and change the state of the worker. The parameter server itself is implemented as an actor, which contains the methods push and pull. We create a parameter server with some initial weights and define a worker task, which takes a parameter server as an argument and submits tasks to it. In the case of given the current weight of the parameter server, alternate training is performed between the computed gradients, and the weight of the parameter server is updated using the generated gradient.

我们使用分布式actor处理器来实现同步和异步参数服对于演员，Ray使用actor来扩展数据流模型。演员本质上是一个有状态的工人（或服务）。当实例化新的actor时，将创建一个新的worker，并且该actor的方法被安排在该特定的worker上，并且可以访问和更改worker的状态。参数服务器本身实现为actor，其中包含push和pull方法。我们创建一个带有一些初始权重的参数服务器并定义一个工作人员任务，它将参数服务器作为参数并向其提交任务。在给定参数服务器的当前权重的情况下，在计算的梯度之间执行替代训练，并且使用生成的梯度更新参数服务器的权重。

In order to clarify the relationship between gradient staleness, learning algorithm and hyper-parameter, We also design a kind of test-bed which can reproduce propagation delay of gradient for analyzing gradient staleness effect on prediction performance. In this simulation environment, it is possible to arbitrarily set the update delay of the gradient in asynchronous parallel machine learning. Figure 4 shows details of the test-bed. The specific process is described below:

At first, a parameter server with some random weights created. And the parameter server will pass the weights to the workers. When the worker updates the weight and pushes back to the parameter server, at this time, it is different from the usual, parameter server does not immediately pass weights to each workers. Instead, it creates a deque to store gradients received from each worker and a timestamp ti is added for each set of gradients. When the worker needs to call the pull function, sets the time at this time to τ0, then scan the entire deque. When τ0 - delay > ti is satisfied, parameter server obtains the gradient corresponding to ti and apply ti and delete ti. Then start the next iteration.

为了阐明梯度过时，学习算法和超参数之间的关系，我们还设计了一种可以重现梯度传播延迟的试验台，用于分析梯度过时效应对预测性能的影响。在该仿真环境中，可以在异步并行机器学习中任意设置梯度的更新延迟。图4显示了试验台的细节。具体过程如下：

首先，创建一些随机权重的参数服务器。参数服务器会将权重传递给工作人员。当工人更新重量并推回参数服务器时，此时，它与通常不同，参数服务器不会立即将权重传递给每个工作人员。相反，它创建一个deque来存储从每个worker接收的渐变，并为每组渐变添加时间戳ti。当工人需要调用拉函数时，将此时的时间设置为τ0，然后扫描整个双端队列。当满足τ0 - delay> ti时，参数服务器获得与ti对应的梯度并应用ti并删除ti。然后开始下一次迭代。

### 4. Experiments 
To confirm the effect of gradient staleness, we performed experiments with several delay settings.
#### 4. 1 Network architecture and Datasets
We chose TensorFlow code implementation on CIFAR10 for our neural network architecture; specifically, we trained CIFAR-10，which is small enough to allow rapid experimental training, but it also has features related to modern networks, including ReLu, and conventional batch normalization, since it gives us more stable learning. We run the image classification problem of cifar-10, which consists of 60,000 32x32 RGB color pictures for a total of 10 categories. There are 50000 training images and 10000 test images(cross-validation). We fixed the minibatch size per worker as 100.

我们在CIFAR10上为我们的神经网络架构选择了TensorFlow代码实现; 具体来说，我们训练了CIFAR-10，它足够小，可以进行快速的实验训练，但它也具有与现代网络相关的功能，包括ReLu和传统的批量标准化，因为它为我们提供了更稳定的学习。 我们运行cifar-10的图像分类问题，它由60,000个32 * 32 RGB彩色图片组成，总共有10个类别。 有50000个训练图像和10000个测试图像（交叉验证）。 我们将每个工人的小批量大小固定为100。

#### 4. 2 Experiments Setting
We test three configurations, namely, sequential mode, synchronous mode, asynchronous mode. And for synchronous mode and asynchronous mode, each of them has two options, the number of workers and delay.

The sequential mode means just one parameter server and one worker to perform parameter exchange. For synchronous mode, we tested one parameter and several workers with no delay. For asynchronous mode, we tested one parameter and several workers with different delay. Note that we did these on a single node.

我们测试三种配置，即顺序模式，同步模式，异步模式。对于同步模式和异步模式，它们中的每一个都有两个选项，即工作者数量和延迟。

顺序模式仅表示一个参数服务器和一个工作程序来执行参数交换。对于同步模式，我们测试了一个参数和几个没有延迟的工人。对于异步模式，我们测试了一个参数和几个具有不同延迟的工作者。请注意，我们在单个节点上执行了这些操作。

In summary, we tested these settings, namely, sequential mode(Sequential), 2 worker’s synchronous modes and 3 worker’s synchronous modes(Sync), and 2 worker’s and 3 worker’s asynchronous mode with no delay(Async delay=0.0), 2 worker’s and 3 worker’s asynchronous mode with with a delay of 0.5 second(Async delay=0.5), 2 worker’s and 3 worker’s asynchronous mode with with a delay of 1 second(Async delay=1.0), 2 worker’s and 3 worker’s asynchronous mode with with a delay of 1.5 second(Async delay=1.5), 2 worker’s and 3 worker’s asynchronous mode with with a delay of 2 second(Async delay=2.0).

Table 1 shows the experiments setups.

总之，我们测试了这些设置，即顺序模式（顺序），2个工作人员的同步模式和3个工作人员的同步模式（同步），2个工人和3个工人的异步模式，没有延迟（异步延迟= 0.0），2个工人和3工人的异步模式，延迟0.5秒（异步延迟= 0.5），2工人和3工人的异步模式，延迟1秒（异步延迟= 1.0），2工人和3工人的异步模式，延迟1.5秒（异步延迟= 1.5），2工人和3工人的异步模式，延迟2秒（异步延迟= 2.0）。

表1显示了实验设置。
#### 4. 3 Results of experiments
Figure 5 shows the result of experiments with 2 workers, and Figure 6 shows the result of experiments with 3 workers. The x-axis shows the number of processed batches. The y-axis shows the loss to perform gradient exchange. We will discuss the results in the next section.

图5显示了2名工人的实验结果，图6显示了3名工人的实验结果。 x轴显示已处理批次的数量。 y轴显示执行梯度交换的损失。 我们将在下一节讨论结果。
#### 4. 4 Discussion on the results 
From this result, it can be seen that the sequential mode is the ideal situation. As the delay grows, the curve becomes more and more unstable, and there are more and more spikes. With the increase of delay, loss becomes more and more difficult to converge. At the same time, we can also see that the synchronous mode and the asynchronous mode with a delay of 0 have basically the same curve.

Compared to the case of two workers, the situation of the three workers has become worse, the training becomes more and more unstable, the spikes are getting more and more, and the loss of the convergence becomes more and more slow to converge. Similar to the result of 2-worker, sequential mode is still the ideal situation. The synchronous mode and asynchronous mode with delay 0 are still basically the same.

从该结果可以看出，顺序模式是理想的情况。 随着延迟的增加，曲线变得越来越不稳定，并且有越来越多的尖峰。 随着延迟的增加，损失变得越来越难以收敛。 同时，我们还可以看到同步模式和延迟为0的异步模式具有基本相同的曲线。

与两名工人的情况相比，三名工人的情况变得更糟，培训变得越来越不稳定，飙升越来越多，收敛的损失变得越来越慢。 类似于2工作者的结果，顺序模式仍然是理想的情况。 具有延迟0的同步模式和异步模式仍然基本相同。

### 5. Conclusion
We have designed a kind of simulation environment which can arbitrarily set the update delay of the gradient in asynchronous parallel machine learning. We have quantitatively measured the loss to perform parameter exchange with several settings; namely, sequential mode, synchronous mode and asynchronous mode with different delay. 

We have revealed that, 1) As delay increases, asynchronous mode becomes more and more unstable, loss becomes very difficult to converge，and the convergence time also increases. 2) As the number of workers increases, the situation becomes more unstable and the loss becomes more difficult to converge. 3) Sequential mode shows the good performance. As a result of experiments with several update delays, we were able to conclude that the influence on the machine learning process due to the expansion of delay.

我们设计了一种可以在异步并行机器学习中任意设置梯度更新延迟的仿真环境。我们已经定量测量了损失，用几种设置进行参数交换;即顺序模式，同步模式和具有不同延迟的异步模式。

我们已经发现，1）随着延迟的增加，异步模式变得越来越不稳定，损耗变得非常难以收敛，并且收敛时间也增加。 2）随着工人数量的增加，情况变得更加不稳定，损失变得更加难以收敛。 3）顺序模式显示出良好的性能。通过多次更新延迟的实验，我们得出结论，由于延迟的扩大，对机器学习过程的影响。

Our future work include the followings:
• Verify that the loss can be made more stable by adjusting hyper-parameters such as learning rate, mini-batch size and regularization parameter or using optimization methods.

• This prototype is not a complete simulated environment, since we use the physical computation time. We will extend this prototype to a fully simulated environment based on discrete event simulation where we can arbitrary set the computation time for more flexible setup.

• Confirm the simulation result with real settings.

我们未来的工作包括以下内容：

•通过调整学习率，小批量大小和正则化参数等超参数或使用优化方法，验证损失是否可以更稳定。

•此原型不是完整的模拟环境，因为我们使用物理计算时间。我们将这个原型扩展到基于离散事件模拟的完全模拟环境，我们可以任意设置计算时间以实现更灵活的设置。

•使用实际设置确认模拟结果。



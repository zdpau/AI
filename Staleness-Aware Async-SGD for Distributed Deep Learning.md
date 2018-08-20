Staleness-Aware Async-SGD for Distributed Deep Learning

https://www.ijcai.org/Proceedings/16/Papers/335.pdf

## abstract
（Deep neural networks have been shown to achieve state-of-the-art performance in several machine learning tasks. Stochastic Gradient Descent (SGD) is the preferred optimization algorithm for training these networks and asynchronous SGD (ASGD) has been widely adopted for accelerating the training of large-scale deep networks in a distributed computing environment. However, in practice it is quite challenging to tune the training hyperparameters(such as learning rate) when using ASGD so as achieve convergence and linear speedup, since the stability of the optimization algorithm is strongly influenced by the asynchronous nature of parameter updates. In this paper, we propose a variant of the ASGD algorithm in which the learning rate is modulated according to the gradient staleness and provide theoretical guarantees for convergence of this algorithm. Experimental verification is performed on commonly-used image classification benchmarks: CIFAR10 and Imagenet to demonstrate the superior effectiveness of the proposed approach, compared to SSGD (Synchronous SGD) and the conventional ASGD algorithm.）

深度神经网络已被证明可以在多个机器学习任务中实现最先进的性能。随机梯度下降（SGD）是用于训练这些网络的优选优化算法，并且异步SGD（ASGD）已被广泛用于加速分布式计算环境中的大规模深度网络的训练。然而，在实践中，在使用ASGD时调整训练超参数（例如学习速率）以实现收敛和线性加速是非常具有挑战性的，因为优化算法的稳定性受到参数更新的异步性质的强烈影响。在本文中，我们提出了ASGD算法的一种变体，其中学习速率根据梯度过时而被调制，并为该算法的收敛提供理论保证。与SSGD（同步SGD）和传统ASGD算法相比，在常用的图像分类基准：CIFAR10和Imagenet上进行了实验验证，以证明所提方法的优越性。

## Introduction
Large-scale deep neural networks training is often constrained by the available computational resources, motivating the development of computing infrastructure designed specifically for accelerating this workload.This includes distributing the training across several commodity CPUs ([Dean et al., 2012],[Chilimbi et al., 2014]),or using heterogeneous computing platforms containing multiple GPUs per computing node ([Seide et al., 2014],[Wu et al., 2015],[Strom, 2015]), or using a CPU-based HPC cluster ([Gupta et al., 2015]).

大规模深度神经网络训练通常受到可用计算资源的限制，促使专门为加速这种工作量而设计的计算基础设施的发展。这包括将培训分布在多个商品CPU上（[Dean et al。，2012]，[Chilimbi et al。，2014]），或者使用每个计算节点包含多个GPU的异构计算平台（[Seide et al。，2014]， [Wu et al。，2015]，[Strom，2015]），或使用基于CPU的HPC集群（[Gupta et al。，2015]）。

Synchronous SGD (SSGD) is the most straightforward distributed implementation of SGD in which the master simply splits the workload amongst the workers at every iteration. Through the use of barrier synchronization, the master ensures that the workers perform gradient computation using the identical set of model parameters. The workers are forced to wait for the slowest one at the end of every iteration. This synchronization cost deteriorates the scalability and runtime performance of the SSGD algorithm. Asynchronous SGD(ASGD) overcomes this drawback by removing any explicit synchronization amongst the workers. However, permitting this asynchronous behavior inevitably adds “staleness” to the system wherein some of the workers compute gradients using model parameters that may be several gradient steps behind the most updated set of model parameters. Thus when fixing the number of iterations, ASGD-trained model tends to be much worse than SSGD-trained model. Further, there is no known principled approach for tuning learning rate in ASGD to effectively counter the effect of stale gradient updates.

同步SGD（SSGD）是SGD最直接的分布式实现，其中主服务器在每次迭代时简单地在工作者之间拆分工作负载。通过使用屏障同步，主控器确保工作人员使用相同的模型参数集执行梯度计算。在每次迭代结束时，工人被迫等待最慢的工人。此同步成本会降低SSGD算法的可伸缩性和运行时性能。异步SGD（ASGD）通过消除工作者之间的任何显式同步来克服这个缺点。然而，允许这种异步行为不可避免地给系统增加了“陈旧性”，其中一些工作人员使用模型参数来计算梯度，该模型参数可以是最新更新的模型参数组之后的几个梯度步骤。因此，在确定迭代次数时，ASGD训练的模型往往比SSGD训练的模型差得多。此外，没有已知的原理方法来调整ASGD中的学习速率以有效地抵消陈旧梯度更新的影响。

Prior theoretical work by [Tsitsiklis et al., 1986] and [Agarwal and Duchi, 2011] [Liu et al., 2013] and recent work by [Liu and Wright, 2015], [Lian et al., 2015], [Zhang et al., 2015] provide theoretical guarantees for convergence of stochastic optimization algorithms in the presence of stale gradient updates for convex optimization and nonconvex optimization, respectively. We find that adopting the approach of scale-out deep learning using ASGD gives rise to complex interdependencies between the training algorithm’s hyperparameters(such as learning rate, mini-batch size) and the distributed implementation’s design choices (such as synchronization protocol, number of learners), ultimately impacting the neural network’s accuracy and the runtime performance. In practice, achieving good model accuracy through distributed training requires a careful selection of the training hyperparameters and much of the prior work cited above lacks enough useful insight to help guide this selection process.

[Tsitsiklis et al。，1986]和[Agarwal and Duchi，2011] [Liu et al。，2013]的先前理论工作和[Liu and Wright，2015]，[Lian et al。，2015]，[ Zhang et al。，2015]分别为存在凸优化和非凸优化的陈旧梯度更新提供了随机优化算法收敛的理论保证。我们发现采用ASGD的横向扩展深度学习方法会导致训练算法的超参数（如学习速率，小批量）和分布式实现的设计选择（如同步协议，学习者数量）之间复杂的相互依赖性。 ），最终影响神经网络的准确性和运行时性能。在实践中，通过分布式培训实现良好的模型准确性需要仔细选择培训超参数，并且上面引用的大部分先前工作缺乏足够的有用见解来帮助指导此选择过程。

The work presented in this paper intends to fill this void by undertaking a study of the interplay between the different design parameters encountered during distributed training of deep neural networks. In particular, we focus our attention on understanding the effect of stale gradient updates during distributed training and developing principled approaches for mitigating these effects. To this end, we introduce a variant of the ASGD algorithm in which we keep track of the staleness associated with each gradient computation and adjust the learning rate on a per-gradient basis by simply dividing the learning rate by the staleness value. The implementation of this algorithm on a CPU-based HPC cluster with fast interconnect is shown to achieve a tight bound on the gradient staleness. We experimentally demonstrate the effectiveness of the proposed staleness-dependent learning rate scheme using commonly-used image classification benchmarks: CIFAR10 and Imagenet and show that this simple, yet effective technique is necessary for achieving good model accuracy during distributed training. Further, we build on the theoretical framework of [Lian et al., 2015] and prove that the convergence rate of the staleness-aware ASGD algorithm is consistent with SGD:O(1/根号t),where T is the number of gradient update steps.

本文提出的工作旨在通过研究深度神经网络分布式训练中遇到的不同设计参数之间的相互作用来填补这一空白。特别是，我们将注意力集中在了解分布式培训期间陈旧梯度更新的影响，并制定减轻这些影响的原则方法。为此，我们引入了ASGD算法的变体，其中我们跟踪与每个梯度计算相关的陈旧性，并通过简单地将学习率除以过时值来基于每个梯度调整学习率。该算法在具有快速互连的基于CPU的HPC集群上的实现被证明可以实现梯度过时的紧密限制。我们通过使用常用的图像分类基准：CIFAR10和Imagenet实验证明了所提出的与陈旧性相关的学习率方案的有效性，并表明这种简单而有效的技术对于在分布式训练期间实现良好的模型准确性是必要的。此外，我们建立在[Lian et al。，2015]的理论框架上，并证明了陈旧性感知ASGD算法的收敛速度与SGD一致：O(1/根号t),其中T是梯度更新步骤的数量。

Previously, [Ho et al., 2013] presented a parameter server based distributed learning system where the staleness in parameter updates is bounded by forcing faster workers to wait for their slower counterparts. Perhaps the most closely related prior work is that of [Chan and Lane, 2014] which presented a multi-GPU system for distributed training of speech CNNs and acknowledge the need to modulate the learning rate in the presence of stale gradients. The authors proposed an exponential penalty for stale gradients and show results for up to 5 learners, without providing any theoretical guarantee of the convergence rate. However, in larger-scale distributed systems, the gradient staleness can assume values up to a few hundreds ([Dean et al., 2012]) and the exponential penalty may reduce the learning rate to an arbitrarily small value, potentially slowing down the convergence. In contrast, in this paper, we formally prove our proposed ASGD algorithm to converge as fast as SSGD. Further, our implementation achieves near-linear speedup while maintaining the optimal model accuracy. We demonstrate this on widely used image classification benchmarks.

以前，[Ho et al。，2013]提出了一种基于参数服务器的分布式学习系统，其中参数更新中的陈旧性受到迫使更快的工人等待其较慢的对应物的限制。也许最密切相关的先前工作是[Chan and Lane，2014]，其提出了用于语音CNN的分布式训练的多GPU系统，并且承认在存在陈旧渐变的情况下调节学习速率的需要。作者提出了对陈旧梯度的指数惩罚，并显示多达5个学习者的结果，而没有提供任何理论上的收敛率保证。然而，在较大规模的分布式系统中，梯度过时可以假设值高达数百（[Dean et al。，2012]），指数惩罚可能会将学习率降低到任意小的值，从而可能减慢收敛速度。相比之下，在本文中，我们正式证明了我们提出的ASGD算法收敛速度与SSGD一样快。此外，我们的实现实现了接近线性的加速，同时保持了最佳的模型精度。我们在广泛使用的图像分类基准上证明了这一点。

## 2 System architecture
In this section we present an overview of our distributed deep learning system and describe the synchronization protocol design. In particular, we introduce the n-softsync protocol which enables a fine-grained control over the upper bound on the gradient staleness in the system. For a complete comparison, we also implemented the Hardsync protocol (aka SSGD) for model accuracy baseline since it generates the most accurate model (when fixing the number of training epochs), albeit at the cost of poor runtime performance.

在本节中，我们将概述分布式深度学习系统并描述同步协议设计。 特别是，我们引入了n-softsync协议，该协议能够对系统中梯度过时的上限进行细粒度控制。 为了进行完整的比较，我们还为模型精度基线实现了Hardsync协议（又名SSGD），因为它生成了最准确的模型（在确定训练时期的数量时），尽管以运行时性能较差为代价。
### 2.1 Architecture Overview
We implement a parameter server based distributed learning system, which is a generalization of Downpour SGD in [Dean et al., 2012], to evaluate the effectiveness of our proposed staleness-dependent learning rate modulation technique. Throughout the paper, we use the following definitions:

我们实现了一个基于参数服务器的分布式学习系统，它是[Dean et al。，2012]中Downpour SGD的推广，用于评估我们提出的与陈旧度相关的学习速率调制技术的有效性。 在整篇论文中，我们使用以下定义：

λ:number of learners (workers).学习者（工人）的数量。

µ: mini-batch size used by each learner to produce　stochastic gradients.每个学习者用来产生随机梯度的小批量大小。

α: learning rate.

Epoch: a pass through the entire training dataset.遍历整个训练数据集。

Timestamp: we use a scalar clock to represent weights timestamp i, starting from i = 0. Each weight update increments the timestamp by 1. The timestamp of a gradient is the same as the timestamp of the weight used to compute the gradient.

我们使用标量时钟来表示权重时间戳i，从i = 0开始。每个权重更新将时间戳递增1.梯度的时间戳与用于计算梯度的权重的时间戳相同。

τi,l:staleness of the gradient from learner l. A learner l pushes gradient with timestamp j to the parameter server of timestamp i, where i >= j. We calculate the staleness τi,l of this gradient as i-j. τi,l>=0 for any i and l.

学习者l将具有时间戳j的梯度推送到时间戳i的参数服务器，其中i> = j。

Each learner performs the following sequence of steps. getMinibatch: Randomly select a mini-batch of examples from the training data; pullWeights: A learner pulls the current set of weights from the parameter server; calcGradient: Compute stochastic gradients for the current mini-batch. We divide the gradients by the mini-batch size; pushGradient: Send the computed gradients to the parameter server; 

每个学习者执行以下一系列步骤。 getMinibatch：从训练数据中随机选择一小批示例; pullWeights：学习者从参数服务器中提取当前权重集; calcGradient：计算当前小批量的随机梯度。 我们将梯度除以小批量大小; pushGradient：将计算的梯度发送到参数服务器;

The parameter server group maintains a global view of the neural network weights and performs the following functions. sumGradients: Receive and accumulate the gradients from the learners; applyUpdate: Multiply the average of accumulated gradient by the learning rate (step length) and update the weights.

参数服务器组维护神经网络权重的全局视图并执行以下功能。 sumGradients：接收并积累学习者的渐变; applyUpdate：将累积梯度的平均值乘以学习速率（步长）并更新权重。
### 2.2 Synchronization protocols 同步协议
We implemented two synchronization protocols: hardsync protocol (aka, SSGD) and n-softsync protocol (aka, ASGD). Hardsync protocol (SSGD) yields a model with the best accuracy number, however synchronization overheads deteriorate the overall runtime performance. n-softsync protocol is our proposed ASGD algorithm that automatically tunes learning rate based on gradient staleness and achieves model accuracy comparable with SSGD while providing a near-linear speedup in runtime.


我们实现了两个同步协议：hardsync协议（aka，SSGD）和n-softsync协议（aka，ASGD）。 Hardsync协议（SSGD）生成具有最佳准确度的模型，但是同步开销会降低整体运行时性能。 n-softsync协议是我们提出的ASGD算法，它基于梯度过时自动调整学习速率，并实现与SSGD相当的模型精度，同时在运行时提供近线性加速。

Hardsync protocol: To advance the weights’ timestamp θ from i to i + 1, each learner l compute a gradient Δθl using a mini-batch size of µ and sends it to the parameter server.
The parameter server averages the gradients over learners and updates the weights according to equation 1, then broadcasts the new weights to all learners. The learners are forced to wait for the updated weights until the parameter server has　received the gradient contribution from all the learners and finished updating the weights. This protocol guarantees that each learner computes gradients on the exactly the same set of weights and ensures that the gradient staleness is 0. The hardsync protocol serves as the baseline, since from the perspective of SGD optimization it is equivalent to SGD using batch size µλ.　公式见论文

Hardsync协议：为了将权重的时间戳θ从i推进到i + 1，每个学习者l使用微批量大小μ计算梯度Δθ1并将其发送到参数服务器。
参数服务器对学习者的渐变进行平均，并根据等式1更新权重，然后将新权重广播给所有学习者。 学习者被迫等待更新的权重，直到参数服务器已经收到来自所有学习者的梯度贡献并完成更新权重。 该协议保证每个学习者在完全相同的权重集上计算梯度并确保梯度过期度为0.硬同步协议用作基线，因为从SGD优化的角度来看，它相当于使用批量大小μλ的SGD。

n-softsync protocol: Each learner l pulls the weights from the parameter server, calculates the gradients and pushes the gradients to the parameter server. The parameter server updates the weights after collecting at least c = (λ/n) gradients from any of the λ learners. Unlike hardsync, there are no explicit synchronization barriers imposed by the parameter server and the learners work asynchronously and independently. The splitting parameter n can vary from 1 to λ.
The n-softsync weight update rule is given by:公式见论文

n-softsync协议：每个学习者从参数服务器中提取权重，计算梯度并将梯度推送到参数服务器。在从任何λ学习者收集至少c =（λ/ n）梯度之后，参数服务器更新权重。与hardsync不同，参数服务器没有明确的同步障碍，学习者可以异步和独立地工作。分裂参数n可以在1到λ之间变化。
n-softsync权重更新规则由下式给出：
### 2.3 Implementation Details
We use MPI as the communication mechanism between learners and parameter servers. Parameter servers are sharded. Each learner and parameter server are 4-way threaded. During the training process, a learner pulls weights from the parameter server, starts training when the weights arrive, and then calculates gradients. Finally it pushes the gradients back to the parameter server before it can pull the weights again. We do not “accrue” gradients at the learner so that each gradient pushed to the parameter server is always calculated out of one mini-batch size as accruing gradients generally lead to a worse model[Lian et al., 2015; Gupta et al., 2015]. In addition, the parameter server communicates with learners via MPI blocking-send calls (i.e., pullWeights and pushGradient), that is the computation on the learner is stalled until the corresponding blocking send call is finished. The design choice is due to the fact that it is difficult to guarantee making progress for MPI nonblocking calls and multi-thread level support to MPI communication is known not to scale [MPI-Forum, 2012]. Further, by using MPI blocking calls, the gradients’ staleness can be effectively bounded, as we demonstrate in Section 2.4. Note that the computation in parameter servers and learners are however concurrent (except for the learner that is communicating with the server, if any). No synchronization is required between learners and no synchronization is required between parameter server shards.

我们使用MPI作为学习者和参数服务器之间的通信机制。参数服务器是分片的。每个学习者和参数服务器都是4向线程。在训练过程中，学习者从参数服务器中提取权重，在权重到达时开始训练，然后计算梯度。最后，它将渐变推回到参数服务器，然后再次拉动重量。我们没有“累积”学习者的渐变，因此推送到参数服务器的每个梯度总是由一个小批量大小计算，因为累积的梯度通常会导致更差的模型[Lian et al。，2015; Gupta等，2015]。另外，参数服务器通过MPI阻塞发送呼叫（即，pullWeights和pushGradient）与学习者通信，即学习者的计算停止，直到相应的阻塞发送呼叫结束。设计选择是由于很难保证MPI非阻塞调用的进展，并且MPI通信的多线程级支持已知不能扩展[MPI-Forum，2012]。此外，通过使用MPI阻塞调用，渐变的陈旧性可以有效地限制，如我们在2.4节中所示。请注意，参数服务器和学习者中的计算是并发的（除了与服务器通信的学习者，如果有的话）。学习者之间不需要同步，参数服务器分片之间不需要同步。

Since memory is abundant on each computing node, our implementation does not split the neural network model across multiple nodes (model parallelism). Instead, depending on the problem size, we pack either 4 or 6 learners on each computing node. Learners operate on homogeneous processors and run at similar speed. In addition, fast interconnect expedites pushing gradients and pulling weights. Both of these hardware aspects help bound (but does not guarantee) gradients’ staleness.

由于每个计算节点上的内存都很丰富，我们的实现不会将神经网络模型分成多个节点（模型并行）。 相反，根据问题的大小，我们在每个计算节点上打包4或6个学习者。 学习者在同类处理器上运行并以相似的速度运行。 此外，快速互连可加速推动梯度和拉动重量。 这两个硬件方面都有助于限制（但不保证）渐变的陈旧性。
### 2.4  Staleness analysis
In the hardsync protocol, the update of weights from θi to θi+1 is computed by aggregating the gradients calculated using weights θi. As a result, each of the gradients gi in the ith step carries with it a staleness τi,l equal to 0.

Figure 1 shows the measured distribution of gradient staleness for different n-softsync (ASGD) protocols when using  λ= 30 learners. For the 1-softsync, the parameter server updates the current set of weights when it has received a total of 30 gradients from (any of) the learners. In this case, the staleness τi,l for the gradients computed by the learner l takes values 0, 1, or 2. Similarly, the 15-softsync protocol forces the parameter server to accumulate λ/15 = 2 gradient contributions from the learners before updating the weights. On the other hand, the parameter server updates the weights after receiving a gradient from any of the learners when the 30-softsync protocol is enforced. The average staleness (τi) for the 15-softsync and 30-softsync protocols remains close to 15 and 30, respectively. Empirically, we have found that a large fraction of the gradients have staleness close to n, and only with a very low probability (< 0.0001) does τ exceed 2n. These measurements show that, in general, τi,l 属于 {0, 1,..., 2n} and (τi)约等于n for the n-softsync protocol. Clearly, the n-softsync protocol provides an effective mechanism for controlling the gradient staleness.

In our implementation, the parameter server uses the staleness information to modulate the learning rate on a pergradient basis. For an incoming gradient with staleness τi,l, the learning rate is set as:

在硬同步协议中，通过聚合使用权重θi计算的梯度来计算从θi到θi+ 1的权重的更新。结果，第i步中的每个梯度gi携带有等于0的陈旧性τi，l。图1示出了当使用λ= 30个学习者时不同的n-softsync（ASGD）协议的梯度陈旧度的测量分布。对于1-softsync，参数服务器
当从（任何）学习者收到总共30个渐变时，更新当前的权重集。在这种情况下，由学习者l计算的梯度的陈旧性τi，l取值0,1或2.类似地，15-softsync协议强制参数服务器累积来自学习者的λ/ 15 = 2梯度贡献更新权重。另一方面，当执行30-softsync协议时，参数服务器在从任何学习者接收到梯度之后更新权重。 15-softsync和30-softsync协议的平均过时（τi）分别接近15和30。根据经验，我们发现大部分梯度具有接近n的陈旧性，并且只有非常低的概率（<0.0001）τ超过2n。这些测量结果表明，对于n-softsync协议，一般来说，τi，l属于{0,1，...，2n}和（τi）约等于n。显然，n-softsync协议提供了一种控制梯度过时的有效机制。
在我们的实现中，参数服务器使用陈旧度信息来调整基于梯度的学习率。 对于具有陈旧性τi，l的传入梯度，学习速率设置为：公式见论文

## 3 Theoretical Analysis
This section provides theoretical analysis of the ASGD algorithm proposed in section 2. More specifically, we will show the convergence rate and how the gradient staleness affects the convergence. In essence, we are solving the following generic optimization problem:公式见论文

还有一堆公式看论文吧

本节提供了第2节中提出的ASGD算法的理论分析。更具体地说，我们将显示收敛速度以及梯度过时如何影响收敛。 实质上，我们正在解决以下通用优化问题：公式什么的看论文。
如果每个学习者一次计算μ梯度并且参数服务器在从学习者接收c个小批量时更新参数，则从参数服务器的角度来看，参数θ的更新过程可以写为

## 4 Experimental Results
### 4.1 Hardware and Benchmark Datasets
We deploy our implementation on a P775 supercomputer. Each node of this system contains four eight-core 3.84 GHz IBM POWER7 processors, one optical connect controller chip and 128 GB of memory. A single node has a theoretical floating point peak performance of 982 Gflop/s, memory bandwidth of 512 GB/s and bi-directional interconnect bandwidth of 192 GB/s.

We present results on two datasets: CIFAR10 and ImageNet. The CIFAR10 [Krizhevsky and Hinton, 2009] dataset comprises of a total of 60,000 RGB images of size 32 ⇥ 32 pixels partitioned into the training set (50,000 images) and the test set (10,000 images). Each image belongs to one of the 10 classes, with 6000 images per class. For this dataset, we construct a deep convolutional neural network(CNN) with 3 convolutional layers each followed by a pooling layer. The output of the 3rd pooling layer connects, via a fully-connected layer, to a 10-way softmax output layer that generates a probability distribution over the 10 output classes. This neural network architecture closely mimics the CIFAR10 model available as a part of the open-source Caffe deep learning package ([Jia et al., 2014]). The total number of trainable parameters in this network are ～90 K (model size of ～350 kB). The neural network is trained using momentum-accelerated mini-batch SGD with a batch size of 128 and momentum set to 0.9. As a data preprocessing step, the per-pixel mean is computed over the entire training dataset and subtracted from the input to the neural network.

For ImageNet [Russakovsky et al., 2015], we consider the image dataset used as a part of the 2012 ImageNet Large Scale Visual Recognition Challenge (ILSVRC 2012). The training set is a subset of the ImageNet database and contains 1.2 million 256⇥256 pixel images. The validation dataset has 50,000 images. Each image maps to one of the 1000 non-overlapping object categories. For this dataset, we consider the neural network architecture introduced in [Krizhevsky et al., 2012] consisting of 5 convolutional layers and 3 fully-connected layers. The last layer outputs the probability distribution over the 1000 object categories. In all, the neural network has ~72 million trainable parameters and the total model size is 289 MB. Similar to the CIFAR10 benchmark, per-pixel mean computed over the entire training dataset is subtracted from the input image feeding into the neural network.

我们在P775超级计算机上部署我们的实现。 该系统的每个节点包含四个八核3.84 GHz IBM POWER7处理器，一个光纤连接控制器芯片和128 GB内存。 单节点的理论浮点峰值性能为982 Gflop / s，内存带宽为512 GB / s，双向互连带宽为192 GB / s。

我们在两个数据集上呈现结果：CIFAR10和ImageNet。 CIFAR10 [Krizhevsky和Hinton，2009]数据集包括总共60,000个尺寸为32×32像素的RGB图像，这些图像被划分为训练集（50,000个图像）和测试集（10,000个图像）。每个图像属于10个类中的一个，每个类有6000个图像。对于这个数据集，我们构建了一个深度卷积神经网络（CNN），其中有3个卷积层，每个卷层后面跟着一个汇集层。第三汇集层的输出通过完全连接的层连接到10路softmax输出层，该输出层在10个输出类上生成概率分布。这种神经网络架构非常类似于CIFAR10模型，可作为开源Caffe深度学习软件包的一部分（[Jia et al。，2014]）。该网络中可训练参数的总数约为90 K（模型大小约为350 kB）。使用动量加速的小批量SGD训练神经网络，批量大小为128，动量设定为0.9。作为数据预处理步骤，在整个训练数据集上计算每像素平均值，并从神经网络的输入中减去每像素平均值。

对于ImageNet [Russakovsky等，2015]，我们将图像数据集视为2012 ImageNet大规模视觉识别挑战（ILSVRC 2012）的一部分。 训练集是ImageNet数据库的子集，包含120万个256×256像素图像。 验证数据集有50,000个图像。 每个图像映射到1000个非重叠对象类别中的一个。 对于该数据集，我们考虑在[Krizhevsky等人，2012]中引入的神经网络结构，其由5个卷积层和3个完全连接的层组成。 最后一层输出1000个对象类别的概率分布。 总之，神经网络具有约7200万个可训练参数，总模型大小为289 MB。 与CIFAR10基准相似，从输入到神经网络的输入图像中减去在整个训练数据集上计算的每像素平均值。
### 4.2 Runtime Evaluation
Figure 2 shows the speedup measured on CIFAR10 and ImageNet, for up to 30 learners. Our implementation achieves 22x-28x speedup for different benchmarks and different batch sizes. On an average, we find that the ASGD runs 50% faster than its SSGD counterpart.

图2显示了在CIFAR10和ImageNet上测量的加速比，最多可供30名学员使用。 我们的实施方案可以针对不同的基准测试和不同的批量大小实现22x-28x的加速。 平均而言，我们发现ASGD的运行速度比其SSGD速度快50％。
### 4.3 Model Accuracy Evaluation
For each of the benchmarks, we perform two sets of experiments:
(a) setting learning rate fixed to the best-known learning rate for SSGD, a = a0, and (b) tuning the learning rate on a per-gradient basis depending on the gradient staleness τ, a = a0/τ . It is important to note that when a = a0 and n = λ(number of learners) in n-softsync protocol, our implementation is equivalent to the Downpour-SGD of [Dean et al., 2012]. Albeit at the cost of poor runtime performance, we also train using the hardsync protocol since it guarantees zero gradient staleness and achieves the best model accuracy. Model trained by Hardsync protocol provides the target model accuracy baseline for ASGD algorithm. Further, we perform distributed training of the neural networks for each of these tasks using the n-softsync protocol for different values of n. This allows us to systematically observe the effect of stale gradients on the convergence properties.

对于每个基准测试，我们执行两组实验：
（a）将学习率设定为SSGD的最佳已知学习率，a = a0，以及（b）根据梯度陈旧性τ，a = a0 /τ，在每个梯度的基础上调整学习率。 值得注意的是，当n-softsync协议中a = a0和n =λ（学习者数量）时，我们的实现等同于[Dean et al。，2012]的Downpour-SGD。 虽然以运行时性能不佳为代价，我们也使用hardsync协议进行训练，因为它可以保证零梯度过时并实现最佳的模型精度。 由Hardsync协议训练的模型为ASGD算法提供目标模型精度基线。 此外，我们使用n-softsync协议针对不同的n值对这些任务中的每一个执行神经网络的分布式训练。 这使我们能够系统地观察陈旧梯度对收敛特性的影响。
### CIFAR10
When using a single learner, the mini-batch size is set to 128 and training for 140 epochs using momentum accelerated SGD (momentum = 0.9) results in a model that achieves ~18% misclassification error rate on the test dataset. The base learning rate a0 is set to 0.001 and reduced by a factor of 10 after the 120th and 130th epoch. In order to achieve comparable model accuracy as the single-learner, we follow the prescription of [Gupta et al., 2015] and reduce the minibatch size per learner as more learners are added to the system in order to keep the product of mini-batch size and number of learners approximately invariant.

Figure 3 shows the training and test error obtained for different synchronization protocols: hardsync and n-softsync,n 属于 (1,　λ )　when using　λ= 30 learners. The mini-batch size per learner is set to 4 and all the other hyperparameters are kept unchanged from the single-learner case. Figure 3 top half shows that as the gradient staleness is increased(achieved by increasing the splitting parameter n in n-softsync protocol), there is a gradual degradation in SGD convergence and the resulting model quality. In the presence of large gradient staleness (such as in 15, and 30-softsync protocols), training fails to converge and the test error stays at 90%. In contrast, Figure 3 bottom half shows that when these experiments are repeated using our proposed stalenessdependent learning rate scheme of Equation 3, the corresponding curves for training and test error for different n-softsync protocols are virtually indistinguishable (see Figure 3 bottom half). Irrespective of the gradient staleness, the trained model achieves a test error of ~18%, showing that **proposed learning rate modulation scheme is effective in bestowing upon the training algorithm a high degree of immunity to the effect of stale gradients.

当使用单个学习者时，小批量大小设置为128，并且使用动量加速SGD（动量= 0.9）训练140个时期导致模型在测试数据集上实现~18％的错误分类错误率。基础学习率a0设定为0.001并且在第120和第130个时期之后减少了10倍。为了达到与单学习者相当的模型精确度，我们遵循[Gupta et al。，2015]的规定，并减少每个学习者的小批量大小，因为更多的学习者被添加到系统中以保持最小的产品批量大小和学习者数量近似不变。

图3显示了使用λ= 30学习者时，针对不同同步协议获得的训练和测试错误：hardsync和n-softsync，n属于（1，λ）。每个学习者的小批量大小设置为4，并且所有其他超参数与单学习者案例保持不变。图3的上半部分显示随着梯度过时性的增加（通过增加n-softsync协议中的分裂参数n来实现），SGD收敛和所得模型质量逐渐降低。在存在大梯度过时（例如15和30-softsync协议）的情况下，训练无法收敛并且测试误差保持在90％。相比之下，图3的下半部分显示，当使用我们提出的方程式3的相关学习速率方案重复这些实验时，不同n-softsync协议的训练和测试误差的相应曲线几乎无法区分（参见图3下半部分）。不考虑梯度过时，训练模型达到约18％的测试误差，**表明所提出的学习速率调制方案有效地赋予训练算法对陈旧梯度的影响的高度免疫性。**
### ImageNet
With a single learner, training with mini-batch size of 256, momentum 0.9 results in top-1 error of 42.56% and top-5 error of 19.18% on the validation set at the end of 35 epochs. The initial learning rate a0 is set to 0.01 and reduced by a factor of 5 after the 20th and again after the 30th epoch. Next, we train the neural network using 18 learners, different n-softsync protocols and reduce the mini-batch size per learner
to 16.

Figure 4 top half shows the training and top-1 validation error when using the learning rate that is the same as the single learner case a0. The convergence properties progressively deteriorate as the gradient staleness increases, failing to converge for 9 and 18-softsync protocols. On the other hand, as shown in Figure 4 bottom half, automatically tuning the learning rate based on the staleness results in nearly identical behavior for all the different synchronization protocols.These results echo the earlier observation that the proposed learning rate strategy is effective in combating the adverse effects of stale gradient updates. Furthermore, adopting the stalenessdependent learning rate helps avoid the laborious manual effort of tuning the learning rate when performing distributed training using ASGD.

**Summary With the knowledge of the initial learning rate for SSGD (a0), our proposed scheme can automatically tune the learning rate so that distributed training using ASGD can achieve accuracy comparable to SSGD while benefiting from near linear-speedup.**

对于单个学习者，小批量训练256的训练，动量0.9导致在35个时期结束时验证集上的前1个误差为42.56％，前5个误差为19.18％。初始学习率a0设定为0.01并且在第20个时期之后并且在第30个时期之后再次减少5倍。接下来，我们使用18个学习者，不同的n-softsync协议训练神经网络，并减少每个学习者的小批量大小
到16岁。

图4上半部分显示了使用与单个学习者案例a0相同的学习率时的训练和前1个验证错误。随着梯度过时性增加，收敛性质逐渐恶化，不能收敛9和18-软同步协议。另一方面，如图4下半部所示，基于陈旧性自动调整学习速率导致所有不同同步协议的行为几乎相同。这些结果与先前的观察结果相呼应，即所提出的学习速率策略在对抗中是有效的。过时梯度更新的不利影响。此外，采用与陈旧度相关的学习率有助于避免在使用ASGD执行分布式培训时调整学习率的费力的手动工作。

**总结根据SSGD（a0）的初始学习速率的知识，我们提出的方案可以自动调整学习速率，使得使用ASGD的分布式训练可以实现与SSGD相当的准确性，同时受益于接近线性加速。**
## 5 Conclusion
In this paper, we study how to effectively counter gradient staleness in a distributed implementation of the ASGD algorithm. We prove that by using our proposed staleness-dependent learning rate scheme, ASGD can converge at the same rate as SSGD. We quantify the distribution of gradient staleness in our framework and demonstrate the effectiveness of the learning rate strategy on standard benchmarks (CIFAR10 and ImageNet). The experimental results show that our implementation achieves close to linear speedup for up to 30 learners while maintaining the same convergence rate in spite of the varying degree of staleness in the system and across vastly different data and model sizes.

在本文中，我们研究如何在ASGD算法的分布式实现中有效地抵消梯度过时。 我们通过使用我们提出的与陈旧性相关的学习率方案证明，ASGD可以以与SSGD相同的速率收敛。 我们量化了框架中梯度陈旧度的分布，并证明了学习率策略在标准基准（CIFAR10和ImageNet）上的有效性。 实验结果表明，尽管系统中的陈旧程度不同，并且数据和模型大小差异很大，但我们的实现可以实现接近30个学习者的线性加速，同时保持相同的收敛速度。

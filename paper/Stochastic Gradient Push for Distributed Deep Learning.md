# Stochastic Gradient Push for Distributed Deep Learning

http://learningsys.org/nips18/assets/papers/63CameraReadySubmissionsgp_FAIR.pdf

## Abstract
Large mini-batch parallel SGD is commonly used for distributed training of deep networks. Approaches that use tightly-coupled exact distributed averaging based on AllReduce are sensitive to slow nodes and high-latency communication. In this work we show the applicability of Stochastic Gradient Push (SGP) for distributed training. SGP uses a gossip algorithm called PushSum for approximate distributed averaging, allowing for much more loosely coupled communications, which can be beneficial in high-latency or high-variability systems. The tradeoff is that approximate distributed averaging injects additional noise in the gradient which can affect the train and test accuracies. We prove that SGP converges to a stationary point of smooth, non-convex objective functions. Furthermore, we validate empirically the potential of SGP. For example, using 32 nodes with 8 GPUs per node to train ResNet-50 on ImageNet, where nodes communicate over 10Gbps Ethernet, SGP completes 90 epochs in around 1.6 hours while AllReduce SGD takes over 5 hours, and the top-1 validation accuracy of SGP remains within 1.2% of that obtained using AllReduce SGD.

大型小批量并行SGD通常用于深度网络的分布式训练。使用基于AllReduce的紧耦合精确分布式平均的方法对慢节点和高延迟通信很敏感。在这项工作中，我们展示了随机梯度推进（SGP）在分布式训练中的适用性。 SGP使用称为PushSum的八卦算法进行近似分布式平均，允许更松散耦合的通信，这在高延迟或高可变性系统中是有益的。权衡（ tradeoff ）是近似分布式平均在梯度中注入额外的噪声，这可能影响训练和测试精度。我们证明SGP收敛于平滑，非凸目标函数的平稳点。此外，我们根据经验验证了SGP的潜力。例如，使用每个节点8个GPU的32个节点在ImageNet上训练ResNet-50，其中节点通过10Gbps以太网进行通信，SGP在大约1.6小时内完成90个时期，而AllReduce SGD需要5个小时，并且SGP的top-1验证准确率保持在使用AllReduce SGD获得的精确度的1.2％之内。

## 1 Introduction
Deep Neural Networks (DNNs) are the state-of-the art machine learning approach in many application areas, including image recognition [4] and natural language processing [15]. Stochastic Gradient Descent (SGD) is the current workhorse for training neural networks. The algorithm optimizes the network parameters, x, to minimize a loss function, f(·), through gradient descent, where the loss function’s gradients are approximated using a subset of training examples (a mini-batch). DNNs often require large amounts of training data and trainable parameters, necessitating non-trivial computational requirements [16, 9]. There is a need for efficient methods to train DNNs in large-scale computing environments.

深度神经网络（DNN）是许多应用领域中最先进的机器学习方法，包括图像识别[4]和自然语言处理[15]。 随机梯度下降（SGD）是目前训练神经网络的主力。该算法优化网络参数x，以通过梯度下降最小化损失函数f（·），其中使用训练示例的子集（小批量）来近似损失函数的梯度。DNN通常需要大量的训练数据和可训练参数，这需要非平凡的计算要求[16，9]。在大规模计算环境中，需要有效的方法来训练DNN。

A data-parallel version of SGD is often adopted for large-scale, distributed training [3, 7]. Worker nodes compute local mini-batch gradients of the loss function on different subsets of the data and then calculate an exact inter-node average gradient using either the ALLREDUCE communication primitive, in synchronous implementations [3], or using a central parameter server, in asynchronous implementations [2]. Using a parameter server to aggregate gradients introduces a potential bottleneck and a central point of failure [8]. The ALLREDUCE primitive computes the exact average gradient at all workers in a decentralized manner, avoiding issues associated with centralized communication and computation.

SGD的数据并行版本通常用于大规模的分布式培训[3,7]。工作节点计算数据的不同子集上的损失函数的局部小批量梯度，然后在同步实现中使用ALLREDUCE通信原语[3]或在异步实现中使用中央参数服务器来计算精确的节点间平均梯度[2]。使用参数服务器聚集梯度会引入潜在的瓶颈和中心故障点[8]。ALLREDUCE原语以分散的方式计算所有worker的精确平均梯度，避免了与集中式通信和计算相关的问题。

However, exact averaging algorithms like ALLREDUCE are not robust in high-latency or highvariability platforms, e.g., where the network bandwidth may be a significant bottleneck, because they involve tightly-coupled, blocking communication (i.e., the call does not return until all nodes have finished aggregating). Moreover, aggregating gradients across all the nodes in the network can introduce non-trivial computational overhead when there are many nodes, or when the gradients themselves are large. This issue motivates the investigation of a decentralized and inexact version of SGD to reduce the overhead associated with distributed training.

然而，像ALLREDUCE这样的精确平均算法在高延迟或高可变性平台中并不健壮，例如，网络带宽可能是一个重要的瓶颈，因为它们涉及紧密耦合，阻塞通信（即，直到所有节点完成聚合，调用才返回）。此外，当存在许多节点时，或者当梯度本身很大时，聚合网络中所有节点上的梯度会带来非平凡的计算开销。这一问题激发了对分散和不精确的SGD版本的调查，以减少与分布式训练相关的开销。

Numerous decentralized optimization algorithms have been studied in the control-systems literature that leverage consensus-based approaches for the computation of aggregate information; see the survey [11] and references therein. Rather than exactly aggregating gradients (as with ALLREDUCE), this line of work uses less-coupled message passing algorithms that compute inexact distributed averages.

在控制-系统control-systems文献中已经研究了许多分散优化算法，这些算法利用基于共识consensus-based的方法来计算聚合信息; 参见调查[11]和其中的参考文献。 这一系列工作使用较少耦合的消息传递算法来计算不精确的分布式平均值，而不是精确地聚合梯度（与ALLREDUCE一样）。

Most previous work in this area has focused on theoretical convergence analysis assuming convex objectives. Recent work has begun to investigate their applicability to large-scale training of DNNs [8,5]. However, these papers study methods based on communication patterns which are static (the same at every iteration) and symmetric (if i sends to j, then i must also receive from j before proceeding). Such methods inherently require blocking communication overhead. State-of-the-art consensus optimization methods build on the PUSHSUM algorithm for approximate distributed averaging [6, 11], which allows for non-blocking, time-varying, and directed (asymmetric) communication. Since SGD already uses stochastic mini-batches, the hope is that an inexact average mini-batch will be as useful as the exact one if the averaging error is sufficiently small relative to the variability in the gradient.

此前该领域的大多数工作都集中在假设凸目标的理论收敛分析上。最近的工作已经开始研究它们对大规模DNN培训的适用性[8,5]。然而，这些论文研究基于静态（在每次迭代时相同）和对称（如果i发送给j，那么i在继续之前也必须从j接收）的通信模式的方法。这些方法本质上需要阻塞通信开销。最先进的一致优化方法建立在PUSHSUM算法的基础上，用于近似分布式平均[6，11]，它允许非阻塞、时变和有向（非对称）通信。由于SGD已经使用了随机小批量，因此希望如果平均误差相对于梯度的变化足够小，则不精确的平均小批量将与精确的小批量一样有用。

This paper studies the use of Stochastic Gradient Push (SGP), an algorithm blending SGD and PUSHSUM, for distributed training of deep neural networks. We provide a theoretical analysis of SGP, showing it converges for smooth non-convex objectives. We also evaluate SGP experimentally, training ResNets on ImageNet using up to 32 nodes, each with 8 GPUs (i.e., 256 GPUs in total).

本文研究了随机梯度推（SGP）（一种融合SGD和PUSHSUM的算法）在深层神经网络分布式训练中的应用。我们提供了对SGP的理论分析，表明它收敛于平滑的非凸目标。 我们还通过实验评估SGP，使用多达32个节点在ImageNet上训练ResNets，每个节点具有8个GPU（即总共256个GPU）。

Our main contributions are summarized as follows. We provide the first convergence analysis for Stochastic Gradient Push when the objective function is smooth and non-convex. We show that, for an appropriate choice of the step size, SGP converges to a stationary point at a rate of O(1/√nK), where n is the number of nodes and K is the number of iterations. In a high-latency scenario, where nodes communicate over 10Gbps Ethernet, SGP runs up to 3× faster than ALLREDUCE SGD and exhibits 88.6% scaling efficiency over the range from 4–32 nodes. The top-1 validation accuracy of SGP matches that of ALLREDUCE SGD for up to 8 nodes (64 GPUs), and remains within 1.2% of ALLREDUCE SGD for larger networks. In comparison to other decentralized consensus-based approaches that require symmetric messaging, SGP runs faster and it produces models with better validation accuracy.

我们的主要贡献总结如下。当目标函数光滑、非凸时，给出了随机梯度推算法的首次收敛性分析。我们证明，对于步长的适当选择，SGP以O（1 /√nK）的速率收敛到静止点，其中n是节点数，K是迭代次数。在节点通过10Gbps以太网进行通信的高延迟场景中，SGP的运行速度比ALLREDUCE SGD快3倍，在4-32个节点范围内的扩展效率达到88.6％。对于最多8个节点（64个GPU），SGP的top-1验证精度与ALLREDUCE SGD的验证精度相匹配，对于较大的网络，SGP的验证精度保持在ALLREDUCE SGD的1.2%内。与需要对称消息传递的其他分散的基于共识的方法decentralized consensus-based approaches相比，SGP运行速度更快，并且生成的模型具有更好的验证准确性。

## 2 Preliminaries 初步措施, 初步行动
**Problem formulation.** We consider the setting where a network of n nodes cooperates to solve the stochastic consensus optimization problem 问题的表述。 我们考虑n个节点的网络协作以解决随机共识优化问题的设置

Each node has local data following a distribution Di, and the nodes wish to cooperate to find the parameters x of a DNN that minimizes the average loss with respect to their data, where Fi is the loss function at node i. Moreover, the goal codified in the constraints is for the nodes to reach agreement(i.e., consensus) on the solution they report. We assume that nodes can locally evaluate stochastic gradients ∇F(xi; ξi), ξi ∼ Di, but they must communicate to access information about the objective functions at other nodes.

每个节点具有遵循分布Di的本地数据，并且节点希望合作找到DNN的参数x，该参数x使关于其数据的平均损失最小化，其中Fi是节点i处的损失函数。此外，约束中编码的目标是让节点就它们报告的解决方案达成协议（即，共识）。我们假设节点可以局部地评估随机梯度∇F（xi;ξi），ξi~Di，但是它们必须进行通信以访问关于其他节点处的目标函数的信息。
## 3 Stochastic Gradient Push
## 4 Experiments
## 5 Conclusion

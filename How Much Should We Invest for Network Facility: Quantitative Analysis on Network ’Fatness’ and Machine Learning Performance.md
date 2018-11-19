http://commondatastorage.googleapis.com/data-dir/mlsys17duo.pdf

## How Much Should We Invest for Network Facility: Quantitative Analysis on Network ’Fatness’ and Machine Learning Performance 我们应该为网络设施投入多少：网络“肥胖”和机器学习绩效的定量分析

### Abstract
Multi-node execution is becoming more and more popular for machine learning because of it’s huge amount of computation. The question we are trying to answer here is that, how should we design computer systems for deep learning, especially in terms of investment for the network. Traditional cluster based ’supercomputers’ require huge amount of investment on network switches since the network ’fatness’ is quite important for the typical applications of super-computers. Do the machine learning workloads share the characteristics with such kinds of applications? To answer this questions, we quantitatively analyze the impact of network fatness on several type of machine learning application types with several network configurations. The results we obtained strongly implies that the network fatness is not important for machine learning applications, and thus we could safely reduce investment on network facilities.

由于计算量巨大，多节点执行在机器学习中变得越来越流行。 我们在这里要回答的问题是，我们应该如何设计用于深度学习的计算机系统，尤其是在网络投资方面。 传统的基于集群的“超级计算机”需要对网络交换机进行大量投资，因为网络“fatness”对于超级计算机的典型应用非常重要。 机器学习工作负载是否与这类应用程序共享特征？ 为了回答这个问题，我们定量分析了网络胖度对几种网络配置的几种机器学习应用类型的影响。 我们获得的结果强烈暗示网络胖度对于机器学习应用程序并不重要，因此我们可以安全地减少对网络设施的投资。

### 1，Introduction
Cluster computers for machine learning are getting popular, along with recent rapid development of deep learning frameworks that support distributed learning. Typical configuration of these cluster with more than 100 nodes requires tree like hierarchical network structure since single switch cannot handle that much nodes. However, simple hierarchical tree structure is notorious for performance degradation for specific kinds of applications. This problem is widely recognized in the HPC community.

用于机器学习的集群计算机越来越受欢迎，以及最近支持分布式学习的深度学习框架的快速发展。具有超过100个节点的这些群集的典型配置需要树状分层网络结构，因为单个交换机不能处理那么多节点。但是，简单的分层树结构因特定类型的应用程序的性能下降而臭名昭着。这个问题在HPC社区中得到广泛认可。

Bisection bandwidth, which is defined as the bandwidth available between two partitions, is one of the measure for network structure in the HPC community. Recently, a network structure called Clos is widely employed as the large scale cluster network. Clos is designed to preserve the bisection bandwidth with small number of network switches, however, it still cost much to keep high bisection bandwidth. The question here is; how much bisection bandwidth do we really need for large scale clusters for machine learning? Do we have to invest network switches as in the HPC community?

二分带宽（定义为两个分区之间可用的带宽）是HPC社区中网络结构的衡量标准之一。最近，一种名为Clos的网络结构被广泛用作大规模集群网络。 Clos旨在通过少量网络交换机来保持二分带宽，但是，保持高二分带宽仍然需要很多成本。这里的问题是;我们真正需要多少对分带宽用于机器学习的大规模集群？我们是否必须像HPC社区一样投资网络交换机？

To answer this question, we conducted comprehensive simulation study. We tested 2 layered and 3 layered Clos network with several bisection bandwidth setting. For the target application, we assumed data parallel machine learning, which is getting popular these days. In the data parallel machine learning application, each machine learning module exchange gradients to proceed the computation. We assumed 2 exchange methods; centralized server based method and direct exchange method. 

为了回答这个问题，我们进行了全面的模拟研究。我们测试了2层和3层Clos网络，并设置了多个二分带宽。对于目标应用程序，我们假设数据并行机器学习，这些日子越来越受欢迎。在数据并行机器学习应用程序中，每个机器学习模块交换梯度以进行计算。我们假设了2种交换方法;基于集中服务器的方法和直接交换方法。

This paper is a throughly rewritten version of [2], which is non-peer reviewed article in Japanese domestic workshop.
本文是[2]的彻底改写版本，是日本国内研讨会上的非同行评审文章。

In the next section we give the background of the research; data parallel machine learning systems, the simulator we used, and the Clos network. Section 3 describes experimental setup and the result of the experiments. Section 4 gives summary of the paper and the future work.

在下一节中，我们将介绍研究背景; 数据并行机器学习系统，我们使用的模拟器和Clos网络。 第3节描述了实验装置和实验结果。 第4节概述了论文和未来的工作。

### 2 Background
#### 2.1 Parameter Exchange Methods for Large Scale Machine Learning Systems 2.1大规模机器学习系统的参数交换方法
To parallelize machine learning systems, there are two methods; Data Parallel and Model Parallel. While data parallel method simultaneously trains multiple machine learning models synchronizing each other, model parallel parallelize inside a single machine learning model. While these two methods are not exclusive each other and often used complementarily, we focus on data parallel in this paper. Data parallel machine learning methods could be categorized into two types; synchronous methods and asynchronous methods; synchronous means all the machine learning models are strictly becomes same periodically, while asynchronous methods allow slight difference among the models. This paper deals with synchronous methods only. To implement data parallel machine learning systems, the are two methods; parameter server based method and direct communication method.

为了并行化机器学习系统，有两种方法;数据并行和模型并行。虽然数据并行方法同时训练多个机器学习模型彼此同步，但模型并行在单个机器学习模型内并行化。这两种方法并不是互相排斥的，而且经常互补使用，本文中，我们关注的是数据并行。数据并行机器学习方法可以分为两类：同步方法和异步方法;同步意味着所有机器学习模型周期性地严格相同，而异步方法允许模型之间的细微差别。本文仅涉及同步方法。要实现数据并行机器学习系统，有两种方法;基于参数服务器的方法和直接通信方法。

**Parameter Server based Method**
Central server to exchange parameters are often called parameter server [3][4][5]. The left diagram in Figure 1 shows the parameter server based parameter exchange. The workers (machine learning modules) send parameters (or gradients) to the parameter server, the parameter server aggregates the parameter, and send back them to the workers. Often, multiple parameter servers are used to shard the parameters and balance the load; each parameter server is responsible for a subset of parameters.

用于交换参数的中央服务器通常称为参数服务器[3] [4] [5]。图1中的左图显示了基于参数服务器的参数交换。工作人员（机器学习模块）将参数（或梯度）发送到参数服务器，参数服务器聚合参数，并将它们发送回工作人员。通常，多个参数服务器用于分割参数并平衡负载;每个参数服务器负责参数的子集。

There are two options for parameter server placement as shown in Figure 3. The left diagram shows the ’packed’ placement where parameter servers are concentrated to one or few sub-clusters. The right diagram shows the ’distributed’ placement where parameter servers are evenly distributed to all the sub-clusters. Note that the parameter server node in each sub-cluster is selected in round-robin fashion to avoid unnecessary network contention in the upper layer switches.

参数服务器放置有两个选项，如图3所示。左图显示了“打包”放置，其中参数服务器集中到一个或几个子集群。 右图显示了“分布式”放置，其中参数服务器均匀分布到所有子集群。 注意，以循环方式选择每个子集群中的参数服务器节点，以避免上层交换机中不必要的网络争用。

**Direct Exchange Method** It is possible to synchronize the models without using central server by repeating peer-to-peer exchange of parameters [6]. The left diagram in Figure 2 shows the communication with 8 workers. Communication pattern likes this is known as butterfly communication, which is widely used, for example, by the allreduce in MPI[7]. It can exchange information with all the nodes within Log2N steps of communication where N is the number of workers.
通过重复参数的对等交换，可以在不使用中央服务器的情况下同步模型[6]。图2中的左图显示了与8名工作人员的通信。像这样的通信模式被称为蝴蝶通信，其被广泛使用，例如，通过MPI中的allreduce [7]。它可以与Log2N通信步骤中的所有节点交换信息，其中N是工作者的数量。

**Cluster Aware Direct Exchange Method** It is possible to further optimize the butterfly method, given the hierarchical structure. To reduce the inter sub-cluster communication, this method once gather the information inside the sub-clusters to the head nodes of sub-clusters, then perform butterfly among the head nodes of the sub-clusers, and then distribute the exchanged information in each cluster. We call this method layered butterfly. The right diagram in Figure 2 shows the layered butterfly method. This communication pattern requires log2n + log2m + log2n steps where n is the number of nodes per sub-cluster and m is number of sub-clusters. Note that the flat butterfly shown above takes log2N = log2nm = log2n + log2m steps; therefore the layered method requires log2n more steps.

群集感知直接交换方法. 考虑到分层结构，可以进一步优化蝶形方法。为了减少子集群间的通信，该方法一旦将子集群内的信息收集到子集群的头节点，然后在子集群的头节点之间执行蝶形，然后在每个集群中分配交换的信息。我们将此方法称为分层蝴蝶。 图2中的右图显示了分层蝶形方法。 该通信模式需要log2n + log2m + log2n步骤，其中n是每个子集群的节点数，m是子集群的数量。 注意，上面所示的扁平蝶形采用log2N = log2nm = log2n + log2m步长; 因此，分层方法需要log2n更多步骤。

#### 2.2 SimGrid: a Distributed Environment Simulator 
SimGrid[8][9] is a simulation framework for distributed parallel applications. SimGrid is based on a discrete event simulation; it does not perform any real computation / communication. It just estimates times to perform computation / communication based on given parameters and records events like‘ start / end of computation / communication ’. The advantage of this type of simulator is that the simulation cost is relatively small. Even with single node computer, SimGrid can handle several thousands of communicating nodes. To simulate a distributed system in SimGrid, users have to describe platform description and deployment description in XML, and the simulation code in C or C++.

SimGrid [8] [9]是分布式并行应用程序的仿真框架。 SimGrid基于离散事件模拟; 它不执行任何实际的计算/通信。 它只是估计基于给定参数执行计算/通信的时间并记录诸如“计算/通信的开始/结束”之类的事件。 这种模拟器的优点是模拟成本相对较小。 即使使用单节点计算机，SimGrid也可以处理数千个通信节点。 要在SimGrid中模拟分布式系统，用户必须用XML描述平台描述和部署描述，并用C或C ++描述模拟代码。

#### 2.3 Cluster Networks Topologies 群集网络拓扑
Bisection bandwidth and ’full’-bisection One of the widely used metrics to evaluate a network is the Bisection bandwidth, which is defined as the following; if the network is bisected into two partitions, the bisection bandwidth is the bandwidth available between the two partitions[10]. If the bisection bandwidth of a network equals to the total bandwidth of one half of the nodes, we call the network with ’full-bisection’ bandwidth. We introduce the term bisection ratio which is defined as follows.

二分带宽和'全'二分法评估网络的一种广泛使用的度量标准是Bisection带宽，其定义如下： 如果网络被分成两个分区，则二分带宽是两个分区之间可用的带宽[10]。 如果网络的二分带宽等于一半节点的总带宽，我们称网络为“全二分”带宽。 我们引入术语二分比率，其定义如下：

bisection ratio = bisection bandwidth/total bandwidth of one half

The bisection ratio of ’full-bisection’ network is 1.0.

**Clos Network** 

Clos network is a class of network which is originally proposed by Charles Clos in 1953, as a non-blocking network for telephone switching[11]. The core idea is to build a large network using multi-stages of small cross-bar switches.

The term is now used to refer a class of fat-tree network[12], which could be considered as a folded version of the original Clos network. We tested 2 types of Clos networks; 2-layered and 3-layered one.

Clos网络是一种网络，最初由Charles Clos于1953年提出，作为电话交换的非阻塞网络[11]。核心思想是使用多级小型交叉开关构建大型网络。该术语现在用于指代一类胖树网络[12]，它可以被视为原始Clos网络的折叠版本。我们测试了两种类型的Clos网络; 2层和3层。

2-layered Clos is relatively simple[13], as shown in Figure 4. Multiple sub-clusters connected with local-switches are connected by multiple upper layer (right in the figure) root switches to mitigate congestion and improve the bisection bandwidth.

2层Clos相对简单[13]，如图4所示。与本地交换机相连的多个子集群通过多个上层（图中右侧）根交换机连接，以缓解拥塞并提高二分带宽。

In this configuration, number of ports of the switches determines the maximum number of subclusters we can have. With n-node switch, n sub-clusters are the maximum, since all the higher layer switches have to be connected with the all the lower layer switches. Figure 4 shows the configuration with 8-ports switches. The left diagram shows the ’full-bisection’ configuration with 8 port switches and 32 nodes in total. We can configure networks with less bisection bandwidth by reducing the number of upper layer switches.

在此配置中，交换机的端口数决定了我们可以拥有的最大子集群数。对于n节点交换机，n个子集群是最大的，因为所有较高层交换机必须与所有较低层交换机连接。图4显示了具有8端口交换机的配置。左图显示了具有8个端口开关和总共32个节点的“全二分”配置。我们可以通过减少上层交换机的数量来配置具有较少二分带宽的网络。

3-layered Clos is rather complicated[12], as shown in Figure 5. The network is composed of multiple ’pods’ which has local 2-layered network structure in them. The pods are connected by multiple root nodes just like 2-layered case. Number of switch ports determines the network structure. Figure 5 shows 3-layered Clos networks with 4 ports switches.

3层Clos相当复杂[12]，如图5所示。网络由多个“pods”组成，其中包含本地2层网络结构。 pod通过多个根节点连接，就像2层的情况一样。交换机端口数决定了网络结构。图5显示了具有4个端口交换机的3层Clos网络。

With k ports switches, we can have k pods. There are k pods, each pod consists of (k/2)2 servers and 2 layers of k/2 k-port switches. Each edge switch connects to k/2 servers and k/2 aggregation switches. Each aggregation switch connects to k/2 edge and k/2 root switches. There are (k/2)2 k-port root switches, each root switch has one port connected to each of k pods. The i th port of any root switch is connected to pod i such that consecutive ports in the aggregation layer of each pod switch are connected to root switches on (k/2) strides. In general, a 3-layered Clos network built with k-port switches supports k3/4 hosts.

使用k端口交换机，我们可以拥有k pod。有k个pod，每个pod由（k / 2）2个服务器和2层k / 2 k端口交换机组成。 每个边缘交换机连接到k / 2服务器和k / 2聚合交换机。 每个汇聚交换机连接到k / 2边缘和k / 2根交换机。 有（k / 2）2个k端口根交换机，每个根交换机有一个端口连接到每个k个pod。 任何根交换机的第i个端口连接到pod i，使得每个pod交换机的聚合层中的连续端口连接到（k / 2）步幅上的根交换机。 通常，使用k端口交换机构建的3层Clos网络支持k3 / 4主机。

**Comparison of the networks**  Table 1 shows the comparison of the two networks. r denotes the bisection ratio. Figure 6 shows the required number of switches to construct a network with number of nodes specified in x-axis for bisection ratio 1, 1/2, 1/4, and 1/8. Note that number of switches are normalized with the 2×2 switch equivalent number; assuming k port switches could be implemented by (k/2)2 2×2 cross bar switches.

网络比较表1显示了两个网络的比较。 r表示二等分比率。 图6显示了构建网络的所需数量的交换机，其中x轴指定的节点数为二分比率1,1 / 2,1 / 4和1/8。 注意，开关的数量用2×2开关等效数标准化; 假设k端口开关可以通过（k / 2）2 2×2横杆开关实现。

The network resource requirement for 3-layered Clos network is smaller than 2-layered one. 3-layered Clos is considered to be better topology in terms of resource requirement.

3层Clos网络的网络资源要求小于2层网络。 就资源需求而言，3层Clos被认为是更好的拓扑。
### 3, experiment
#### 3.1 Network Setup
We have setup clusters with the two network topologies; 128, 512, and 2048 nodes for 2-layered Clos network, and 256 and 1024 nodes for 3-layered Clos network. We set the bandwidth of links as 4GBytes/s (assuming 40G Infiniband with TCP overhead), and 1GBytes/s (assuming 10G Ethernet with TCP overhead), and the switch latency as 0.2 µs and 1 µs .

我们使用两种网络拓扑建立集群; 用于2层Clos网络的128,512和2048个节点，以及用于3层Clos网络的256和1024个节点。 我们将链路带宽设置为4GBytes / s（假设带有TCP开销的40G Infiniband）和1GBytes / s（假设带有TCP开销的10G以太网），并且切换延迟为0.2μs和1μs。

#### 3.2 Parameter Exchange Methods
We test one parameter server based method and two butterfly based methods. For the parameter server based method, we assume 1/8 of whole nodes in the cluster are used for parameter servers while the others are used for workers. We tested two placement strategy for parameter server based method. One is ’packed’ and the other is ’distributed’, shown in Figure 3.

For the butterfly based methods, all the nodes are used for workers. 2 We test the simple flat-butterfly method with the layered-butterfly method.

我们测试一个基于参数服务器的方法和两个基于蝴蝶的方法 对于基于参数服务器的方法，我们假设集群中1/8的整个节点用于参数服务器，而其他节点用于工作服务器。 我们测试了基于参数服务器方法的两种放置策略。 一个是“打包”，另一个是“分布式”，如图3所示。
对于基于蝴蝶的方法，所有节点都用于工作人员。 我们用分层蝶形方法测试简单的平蝴蝶法。

In summary, we test four settings; namely, parameter server with packed placement (PS, packed) and distributed placement (PS, distributed) , flat butterfly (BF, flat) , and layered butterfly (BF, layered).
#### 3.3 Results and Discussion
Due to space limitation we show some results only here. Note that the results are quite consistent regardless of the size of cluster. Figure 7 shows the result with 2-layered Clos network with 2048 nodes Figure 8 shows the result with 3-layered Clos network with 1024 nodes. The results are for 4GBytes/s bandwidth and 0.2 µs switch latency. The x-axis shows the bisection ratio. The y-axis shows the execution time to perform one gradient exchange.

From this result, it can be seen that the method using the parameter server is inferior to the butterfly network based method in basic performance. This is because the connections to the parameter servers becomes the bottleneck.

由于空间限制，我们仅在此处显示一些结果。 请注意，无论簇的大小如何，结果都非常一致。 图7显示了具有2048个节点的2层Clos网络的结果。图8显示了具有1024个节点的3层Clos网络的结果。 结果是4GBytes / s带宽和0.2μs切换延迟。 x轴表示二等分比率。 y轴显示执行一次梯度交换的执行时间。

从该结果可以看出，使用参数服务器的方法在基本性能方面不如基于蝶形网络的方法。 这是因为与参数服务器的连接成为瓶颈。

备注：Actually, the number of worker nodes is different between butterfly network based method and parameter server based method. Number of worker nodes of parameter server based method is always 1/8 nodes fewer. However, even if the butterfly network based method reduces n nodes, the execution time is expected to be the same.实际上，基于蝶形网络的方法和基于参数服务器的方法之间的工作节点的数量是不同的。 基于参数服务器的方法的工作节点数总是少1/8节点。 然而，即使基于蝶形网络的方法减少n个节点，预期执行时间也是相同的。

Parameter server method with packed placement setting exhibit significant performance drops as the bisection ratio decreases. On the other hand, with distributed placement setting, the parameter server method is hardly affected by the reduction of the bisection bandwidth. This is because of the network traffic is smoothed throughout the cluster by distributing the parameter server nodes.

随着二等分比率降低，具有打包放置设置的参数服务器方法表现出显着的性能下降。另一方面，通过分布式布局设置，参数服务器方法几乎不受二分带宽的减小的影响。这是因为通过分发参数服务器节点在整个群集中平滑网络流量。

Butterfly network based method is faster than parameter server based method, in general. The flat butterfly method tends to be affected by the reduced bisection bandwidth, since it performs inter cluster communication heavily. In contrast, the layered butterfly method is not affected by the bisection bandwidth at all. This result implies that with this method, we do not have to invest in the bisection bandwidth.

基于蝴蝶网络的方法通常比基于参数服务器的方法更快。扁平蝶形方法倾向于受到减小的二分带宽的影响，因为它大量地执行群集间通信。相反，分层蝶形方法完全不受二分带宽的影响。这个结果意味着使用这种方法，我们不必投资于二分带宽。

While it is difficult to directly compare the Clos and Fattree networks, since it is very difficult to setup them with same nodes, they share same trend in results, as shown in Figure 8 and Figure 7. Given that the Clos requires less network resources, we could conclude that Clos network is more suitable for this particular application.

虽然很难直接比较Clos和Fattree网络，但由于使用相同的节点设置它们非常困难，因此它们在结果中共享相同的趋势，如图8和图7所示。鉴于Clos需要较少的网络资源，我们可以得出结论，Clos网络更适合这个特定的应用程序。

Figure 9 shows the result with 1GBytes/s network, instead of 4GBytes/s network in Figure 8. Comparing Figure 9 and Figure 8, we could see that network bandwidth linearly affect the gradient exchange speed. This implies that investing faster network technology will be fruitful.

图9显示了1GBytes / s网络的结果，而不是图8中的4GBytes / s网络。比较图9和图8，我们可以看到网络带宽线性影响梯度交换速度。这意味着投资更快的网络技术将是富有成效的。

Figure 10 shows a close up of Figure 8. As shown in the figure, the flat butterfly is slightly faster than layered butterfly with full-bisection bandwidth. This is because the flat butterfly requires fewer steps than the layered one, as discussed in 2.1.

图10示出了图8的特写。如图所示，扁平蝶形物比具有全分割带宽的分层蝶形物略快。这是因为平面蝴蝶比分层蝴蝶需要更少的步骤，如2.1中所讨论的。
### 4, conclusion
We have quantitatively evaluated the performance of several parameter exchange method for two network topologies; namely, 2-layered and 3-layered Clos networks, with several bisection-ratio to investigate the proper investment on network for distributed machine learning applications. We have revealed that, 1) Bisection ratio affects some of the parameter exchange methods, but cluster aware direct exchange method does not get affected, 2) Network speed linearly affect the parameter exchange speed, 3) Parameter server based methods are substantially slower than the direct exchange methods, 4) Cluster aware direct exchange method (layered butterfly) outperforms naive exchange method (flat butterfly), except for the case with full-bisection bandwidth.

我们定量评估了两种网络拓扑的几种参数交换方法的性能; 即，2层和3层Clos网络，具有几个二分比率，用于研究分布式机器学习应用的网络上的适当投资。 我们已经发现，1）二分比影响了一些参数交换方法，但是群集感知直接交换方法没有受到影响，2）网络速度线性影响参数交换速度，3）基于参数服务器的方法明显慢于 直接交换方法，4）群集感知直接交换方法（分层蝶形）优于天真交换方法（平面蝶形），除了具有全二分带宽的情况。

We conclude that if we employ proper parameter exchange method, we could substantially reduce the investment on the network.
Our future work include the followings:

我们得出结论，如果我们采用适当的参数交换方法，我们可以大大减少对网络的投资。

我们未来的工作包括以下内容：

• Investigate asynchronous gradient exchange setting which will require less network resources.调查异步梯度交换设置，
•使用实际设置确认模拟结果。

• Confirm the simulation result with real settings.这将需要较少的网络资源。



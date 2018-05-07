https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxtbHN5c25pcHMyMDE2fGd4OjRhZDllZDA5MmZhZWRmZGY
## Abstract
#### 深度神经网络的大型数据集的训练时间是深度学习的几个重要应用中的主要瓶颈，例如自动车辆中的对象分类和检测。 为了尽量减少训练时间，必须使用分布式优化方法（例如同步或异步随机梯度下降（SGD））在多个处理节点上对训练进行缩放。虽然同步SGD目前在大规模中展示出了最大吞吐量，但同步scale会因需要在每个梯度步骤同步所有节点而受到影响。 在使用参数服务器的异步方法中，由于争用参数服务器而导致训练速度变慢。 在本文中，我们将同步和异步方法的训练时间与SGD进行比较，以便在ImageNet分类问题上训练现代ResNet体系结构。 为了解决同步和异步方法的缺陷，我们提出了一种异步方法，gossiping SGD，旨在保留这两种方法的积极特性。 我们发现包括弹性平均和闲聊在内的异步SGD在较少节点（最多约32个节点）和较大步长下收敛速度更快，而同步SGD在32个节点上最多可扩展至128个节点。 在128个节点之上获得好的扩展性能仍然是一个悬而未决的问题。
## introduction
#### 自驾车收集的数据估计至少从750 MB / s开始。 通过适当的注释或通过无监督的学习方案，所有这些数据都可用于培训自动驾驶汽车的物体检测系统或电网占用系统。 由此产生的训练集可以在单个CPU / GPU系统上实现数周或更长时间的训练时间。 因此，对于这样的应用程序，训练时间定义了工作流程中最耗时的元素，并且减少训练时间是非常需要的。
#### 为了显着缩短训练时间，培训必须分布在多个CPU / GPU上，实现强大的扩展：随着更多节点（即计算服务器）抛出问题，理想情况下，训练时间应按比例减少。 对于训练深度神经网络，分布式随机梯度下降（SGD）有两种主要方法：（i）基于a fast all-reduce集体通信操作同步all-reduce SGD [1，2，3，4],（ii）使用参数服务器的异步SGD[5，6]。
#### （ⅰ）和（ⅱ）两种方法都存在规模上的弱点。 同步SGD受处理器分散处理，计算资源利用不足，以及面对失败的处理器或节点时不稳健。 另一方面，使用参数服务器的异步方法会造成通信瓶颈，并且可用网络资源不足，从而减慢收敛速度。个人研究人员也可以使用多个节点，包括计算和网络资源。 因此，对于具有有限资源的从业人员来说，确定给定数量节点的最佳方法以及扩展到最多节点的方法是有意义的。
#### 我们关心的是如何缩短培训深度神经网络的时间。 实际上，这转化为将培训扩展到许多处理器的问题。 我们主要关注整体收敛时间; 然而，收敛速度随训练运行而变化，并且还取决于节点的数量。 因此我们调查了以下两个问题：1，（a）异步和同步SGD算法在训练开始阶段（大步长）和训练结束时（小步长）的收敛速度有多快？（b）SGD的收敛性如何随着节点的数量而变化？
#### 为了比较异步和同步SGD算法的优缺点，我们使用各种分布式SGD方法在ImageNet [8]上训练现代ResNet卷积网络[7]。我们主要比较同步all-reduce SGD，最近提出的异步弹性平均（elastic averaging）SGD [9]，以及我们自己的方法，基于最初在不同问题设置中开发的算法[10]，异步gossiping SGD。闲话SGD是一种异步方法，不使用集中式参数服务器;闲谈可以被认为是一种分散式的弹性平均。我们发现包括弹性平均和闲聊在内的异步SGD在较大步长下表现出最好的缩放比例，并且在小规模（最多约32个节点）时可能违反直觉。对于更小的步长和更大的规模，all-reduce会比异步方法更快地收敛到最准确的解决方案。
## background
#### 我们将对SGD使用以下命名约定：θ是最小化目标的参数，θ撇是中心参数（如果适用），α是步长，μ是动量，下标i指的是p个节点中的第i个节点，下标t指的是第t个（小批次）迭代。此外，b指的是每个节点的minibatch大小，而m指的是所有节点上汇总的minibatch大小总和。
### Synchronous All-Reduce SGD
#### 在同步all-reduce SGD中，两个阶段在锁定步骤中交替进行：（1）每个节点计算其局部参数梯度，以及（2）所有节点共同通信以计算聚集梯度，就好像它们都形成了大分布式小型块。聚合梯度的第二阶段形成一个障碍并且是通信密集阶段，通常通过同名的all-reduce操作来实施。all-reduction的时间复杂度可以分解为延迟限制和带宽限制条件。尽管延迟项与O（log（p））成比例，但是有快速环算法，它们的带宽项独立于p [11]。现代网络能够处理1-10GB / s数量级的带宽，并结合10-100 MB数量级的神经网络参数，网络中节点之间的梯度或参数通信可以非常快速。相反，all-reduce的通信开销是由于同步障碍造成的：每个节点必须等待所有其他节点完成all-reduce才能继续进行下一次minibatch迭代。这导致了最慢的节点会阻止剩下的节点取得进展。在[1]，[12]，[2]，[3]和[4]中给出了用于分布式深度学习的大规模同步数据并行SGD示例。我们在算法1中为同步数据并行SGD提供伪代码。
### Asynchronous Parameter-Server SGD
#### 不同的SGD方法包括每个节点异步执行其自己的渐变更新，并偶尔将其参数与中央参数存储同步。这种异步SGD的形式被“Hogwild”SGD [13]推广，它考虑解决单机共享内存系统中的稀疏问题。 Downpour SGD [5]推广了分布式SGD方法，其中节点与中心参数服务器通信梯度。对于SGD的异步参数服务器方法的主要弱点是worker与中央服务器all-to-one，并且通信吞吐量受到服务器上的有限连接接收带宽的限制。缓解通信瓶颈的一种方法是在两轮通信之间引入延迟，但增加延迟会大大降低收敛速度[9]。谷歌DistBelief [5]和微软Adam [6]系统开创了大规模异步SGD用于深度学习。非深度学习环境中的大型参数服务器系统也已在[14]和[15]中得到证明。
### Elastic Averaging SGD
#### 弹性平均SGD [9]是最近提出的算法，属于异步参数服务器方法族，它对通常的随机梯度目标进行了修改，以实现更快的收敛。 除了损失之外，弹性平均寻求最大化中心参数θ撇与局部参数θi之间的一致性：算法2给出了弹性平均算法，其中Xi是节点i的数据，ρ是一致项（consensus objective）的超参数。弹性平均的一致性目标与ADMM的增广拉格朗日密切相关，由一致性目标导出的梯度更新由[9 ]实验证明明显快于标准参数服务器异步SGD。但是，由于弹性平均是异步参数服务器方法族中的一员，它仍然受到中央服务器和客户端工作人员之间的通信瓶颈的影响。（公式见论文）
#### 由于最近公布的结果表明，弹性平均主导了先前的异步参数服务器方法[9]，因此我们将只考虑弹性平均。
## Gossiping SGD
#### 简而言之，同步all-reduce算法由两个重复阶段组成：（1）计算每个节点处的局部梯度，以及（2）通过all-reduce准确聚合局部梯度。为了获得gossip  SGD，我们希望用更加异步友好的通信模式来取代同步all-reduce操作。我们使用的基本构建模块是一种gossip聚合算法[16,17]，它与SGD结合导致gossip SGD算法。在[10]中引入了异步gossip SGD，以用于节点之间的稀疏通信图（例如无线传感器网络）的一般情况。gossip的原始问题设置通常也涉及同步通信，而我们最感兴趣的是异步gossip。
#### 我们还可以通过从概念上将gossiping和弹性平均相结合来推导gossiping SGD更新的数学表达式。我们引入全球一致目标的分布式版本，其中中心参数被本地参数的平均值替代：（公式见论文）
#### 如果jt，i如上所述统一选择，那么该算法等同于“pull-gossip”，即每个节点每次迭代从一个且仅一个其他随机节点拉或接收θj。 另一方面，如果我们用多个节点查询θj代替“单节点估计量”，并且约束条件是每个迭代每个j只表示一次，那么算法就变成“push-gossip”,即每个节点将其自己的θi推送或发送给一个且仅一个其他随机节点，同时从零与多个其他节点之间接收。 Push-Gossiping SGD可以被解释为一个梯度步骤和一个简单的push-sum八卦步骤[16]的交错。 算法3和4分别描述pull-gossiping and push-gossiping SGD。
## experiment
### Implementation
#### 我们使用消息传递接口（MPI）[18]实现了闲谈SGD和其他算法的通信系统。 因为我们想在集群计算环境中使用Infiniband或更专业的互连来运行我们的代码，那么针对MPI是最简单的解决方案。 我们将代码运行在GPU上，使用Nvidia CUDA 7.0驱动程序，并使用cuBLAS和cuDNNv4 [19]库作为核心计算内核。
#### 对于p = 16节点的实验，我们使用由16台机器组成的本地集群，每台机器由Nvidia Kepler K80双GPU，8核英特尔Haswell E5-1680v2 CPU和Mellanox ConnectX-3 FDR 4× Infiniband（56 Gb / s）NIC。我们每K80只使用一个GPU。
#### 对于p = 128个节点的较大规模实验，我们使用了一台GPU超级计算机，其中有超过10,000个节点[20]。 节点包括Nvidia Kepler K20X GPU和8核AMD Bulldozer Opteron 6274 CPU，并通过Cray Gemini互连以3D环面配置连接。
### Methodology（方法）
#### 我们选择ResNets [7]作为我们的神经网络体系结构; 具体来说，我们训练了ResNet-18，它足够小，可以快速训练，但也具有与现代网络相关的特征，包括深度，剩余层数和批量归一化[21]。 我们运行ImageNet的图像分类问题，其中包含128万幅训练图像和50,000幅验证图像，分为1000个类别[8]。 我们的数据增强如下：我们通过将图像的最短维度缩放到256和480像素之间来执行多尺度训练[22]，我们随机选择了224×224个作物和水平翻转，并且我们添加了像素方式的色噪声[23]。 我们评估验证集合图像中心作物的验证损失和top-1错误，最短尺寸缩放为256像素。
#### 除非另有说明，否则我们将步长初始化为α= 0.1，然后将其退火（anneal 有锻炼的意思）两次，每次退火0.1倍。 对于我们的总体小批量大小为m = pb = 256的实验，我们准确地将150k和300k迭代训练退火（有关批量大小设置的详细信息，请参阅第4.3节）。对于我们用较大聚合小批量大小的实验，我们减少了步长退火的迭代次数。 我们使用μ= 0.9的Nesterov动量[24]和λ= 10-4的权重衰减。 对于全部减少和闲聊，我们使用了通信间隔τ= 1，即每次迭代发生通信。 对于弹性平均，我们设定β= 0.8 / p，同时使用τ= 1和τ= 10（后者在[9]中推荐）。
### result
#### 我们的第一组实验比较了在p = 8和p = 16时的all-reduce，弹性平均和push-gossip，并且总体小批量大小为m = pb = 256。结果如图1所示。对于p = 8，弹性平均 通信延迟τ= 10比其他方法更快地完成每次迭代。 有趣的是，在p = 8时，all-reduce在系统上实际上没有同步开销，并且和闲聊一样快。 对于p = 16，闲聊收敛速度快于弹性平均，τ= 10，并且都领先all-reduce。 另外，一旦步长退化到一个很小的值（在这种情况下α= 0.001），同时具有τ= 1和τ= 10的弹性平均就会遇到与其他方法相同的验证损失。
#### 我们在GPU超级计算机上的p = 32个节点，p = 64个节点和p = 128个节点上执行更大规模的实验。 在超级计算机环境中，弹性平均和push-gossiping与我们用来实现算法的MPI的远程存储器接口不匹配，因此我们只显示同步SGD和pull-gossip SGD的结果。结果如图2所示。在这种规模下，我们开始看到同步all-reduce SGD的缩放优势。一次迭代的gossip SGD仍然比一次迭代的all-reduce SGD的迭代速度更快，并且闲聊在初始步长时很快就会工作。但是在步骤大小退火之后，闲聊SGD开始收敛得慢得多。
#### 我们注意到SGD的培训时间可以被认为是product（每次迭代的挂钟时间）×（迭代次数）。我们做了一个与[4]一致的观察：让同步all-reduceSGD运行很多epochs，它通常会收敛到比弹性平均或闲聊SGD更低的最佳验证损失。 我们发现让all-reduce SGD运行超过100万次迭代，并且小批次大小为256，导致68.7％（或31.3％top-1 错误）的峰值top-1验证准确度。然而，使用p = 16个节点时，弹性平均常常难以达到67％的准确率，就像使用多于p = 32的闲聊一样。换句话说，尽管每次迭代的壁钟时间较低，但异步方法需要更多迭代才能收敛。
## discussion
#### 我们感兴趣的是如何获得训练深度神经网络的最短时间。 实际上，这意味着扩展到多个节点。 由于收敛速度随迭代次数以及节点数量而变化，我们提出了以下问题：（a）异步和同步SGD算法在训练开始阶段（大步长）和训练结束时（小步长）的收敛速度有多快？
#### 多达32个节点，当步长较大时，异步SGD可以比all-reduce SGD更少的时间收敛到给定的精度水平。在步长较小（α= 0.001）时，闲聊可以比弹性平均更快地收敛，但all-reduce SGD收敛最为一致。
#### （b）SGD的收敛性如何随着节点的数量而变化？
#### up to 16-32节点，弹性平均和闲话似乎比同步all-reduce SGD收敛速度更快。多达128个节点，all-reduce SGD可以始终如一地收敛到高精度解决方案，而异步方法稳定在较低精度。 特别是，gossip SGD并不像以及具有更多节点的同步SGD那样的事实表明，不同步和通信模式负责收敛的差异，而不是通信量（这两种方法的通信量都很低）。
#### 我们注意到scaling是其他减少训练时间的方法的补充，如设计新网络[25]和量化梯度[12]。在我们的调查中，我们发现异步方法在8,16和32个节点上速度最快。
#### 在32个节点以上，同步SGD提供最快的培训。 这有点违反直觉，因为人们可能自然地认为在同步SGD期间等待在零散节点上的惩罚会随着节点数量的增加而增加，并且由此导致同步SGD不能很好地扩展。 最后，在128个节点之上获得好的缩放结果仍然是一个挑战。


### 图1：在训练挂钟时间和epoch时，ImageNet上的center-crop验证损失和top-1错误。 显示的是：（左）p = 8个节点，每节点minibatch大小b = 32，（右）p = 16个节点，每节点minibatch大小b = 16。
### 图2：在训练挂钟时间和epoch时，ImageNet上的center-crop验证损失和top-1错误，不同节点数p和每节点minibatch大小b = 16。所示为：（左）p = 32个节点，（中 ）p = 64个节点，（右）p = 128个节点。
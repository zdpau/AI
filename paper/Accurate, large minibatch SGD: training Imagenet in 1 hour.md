Accurate, large minibatch SGD: training Imagenet in 1 hour

https://arxiv.org/pdf/1706.02677.pdf

## Abstract
Deep learning thrives with large neural networks and large datasets. However, larger networks and larger datasets result in longer training times that impede research and development progress. Distributed synchronous SGD offers a potential solution to this problem by dividing SGD minibatches over a pool of parallel workers. Yet to make this scheme efficient, the per-worker workload must be large, which implies nontrivial growth in the SGD minibatch size. In this paper, we empirically show that on the ImageNet dataset large minibatches cause optimization difficulties, but when these are addressed the trained networks exhibit good generalization. Specifically, we show no loss of accuracy when training with large minibatch sizes up to 8192 images. To achieve this result, we adopt a hyperparameter-free linear scaling rule for adjusting learning rates as a function of minibatch size and develop a new warmup scheme that overcomes optimization challenges early in training. With these simple techniques, our Caffe2-based system trains ResNet-50 with a minibatch size of 8192 on 256 GPUs in one hour, while matching small minibatch accuracy. Using commodity hardware, our implementation achieves ∼90% scaling efficiency when moving from 8 to 256 GPUs. Our findings enable training visual recognition models on internet-scale data with high efficiency.

深度学习通过大型神经网络和大型数据集而蓬勃发展。但是，较大的网络和较大的数据集会导致较长的培训时间，从而阻碍研究和开发进度。分布式同步SGD通过将SGD小批量分配给并行工作池来提供潜在的解决方案。然而，为了使该方案有效，每个工人的工作量必须很大，这意味着SGD小批量大小的非常大的增长。在本文中，我们凭经验表明，在ImageNet数据集上，大型微型计算机会导致优化困难，但是当这些问题得到解决时，经过训练的网络会表现出良好的概括性。具体而言，我们在使用大型8到2张图像的大型小批量培训时，不会显示精度损失。为了实现这一结果，我们采用了一种超参数线性缩放规则，用于根据小批量大小调整学习率，并开发一种新的预热方案，以便在培训早期克服优化挑战。通过这些简单的技术，我们基于Caffe2的系统可以在一小时内对256个GPU上的小批量8192的ResNet-50进行训练，同时匹配小的小批量精度。使用商用硬件，我们的实现在从8个GPU移动到256个GPU时实现了约90％的扩展效率。我们的研究结果使得能够高效地在互联网规模数据上培训视觉识别模型。

## 1,Introduction
Scale matters. We are in an unprecedented era in AI research history in which the increasing data and model scale is rapidly improving accuracy in computer vision[22, 41, 34, 35, 36, 16], speech [17, 40], and natural language processing [7, 38]. Take the profound impact in computer vision as an example: visual representations learned by deep convolutional neural networks [23, 22] show excellent performance on previously challenging tasks like ImageNet classification [33] and can be transferred to difficult perception problems such as object detection and segmentation [8, 10, 28]. Moreover, this pattern generalizes: larger datasets and neural network architectures consistently yield improved accuracy across all tasks that benefit from pretraining[22, 41, 34, 35, 36, 16]. But as model and data scale grow, so does training time; discovering the potential and limits of large-scale deep learning requires developing novel techniques to keep training time manageable.

规模很重要。 我们正处于人工智能研究历史上前所未有的时代，其中不断增长的数据和模型规模正在迅速提高计算机视觉的准确性[22,41,34,35,36,16]，语音[17,40]和自然语言处理 [7,38]。 以计算机视觉中的深远影响为例：深度卷积神经网络[23,22]所学的视觉表征在以前具有挑战性的任务中表现出色，如ImageNet分类[33]，并且可以转移到难以理解的问题，如物体检测和 分割[8,10,28]。 此外，这种模式概括：较大的数据集和神经网络架构始终在所有受益于预训练的任务中提高准确性[22,41,34,35,36,16]。 但随着模型和数据规模的增长，培训时间也在增长; 发现大规模深度学习的潜力和局限需要开发新技术以保持训练时间的可控性。


The goal of this report is to demonstrate the feasibility of, and to communicate a practical guide to, large-scale training with distributed synchronous stochastic gradient descent(SGD). As an example, we scale ResNet-50 [16] training, originally performed with a minibatch size of 256 images(using 8 Tesla P100 GPUs, training time is 29 hours), to larger minibatches (see Figure 1). In particular, we show that with a large minibatch size of 8192, we can train ResNet-50 in 1 hour using 256 GPUs while maintaining the same level of accuracy as the 256 minibatch baseline.
While distributed synchronous SGD is now commonplace, no existing results show that generalization accuracy can be maintained with minibatches as large as 8192 or that such high-accuracy models can be trained in such short time.

本报告的目的是证明分布式同步随机梯度下降（SGD）的大规模训练的可行性，并传达实用指南。 作为一个例子，我们扩展ResNet-50 [16]训练，最初使用256个图像的小批量（使用8个特斯拉P100 GPU，训练时间为29小时）到更大的小型游艇（见图1）。 特别是，我们表明，对于8192的大型小批量，我们可以使用256个GPU在1小时内训练ResNet-50，同时保持与256个小批量基线相同的准确度。
虽然分布式同步SGD现在司空见惯，但现有结果并未显示通过8192的小型机可以保持通用精度，或者可以在如此短的时间内训练这种高精度模型。

To tackle this unusually large minibatch size, we employ a simple and hyperparameter-free linear scaling rule to adjust the learning rate. While this guideline is found in earlier work [21, 4], its empirical limits are not well understood and informally we have found that it is not widely known to the research community. To successfully apply this rule, we present a new warmup strategy, i.e., a strategy of using lower learning rates at the start of training [16], to overcome early optimization difficulties. Importantly, not only does our approach match the baseline validation error, but also yields training error curves that closely match the small minibatch baseline. Details are presented in §2.

为了解决这种异常大的小批量大小问题，我们采用了一种简单且无超参数的线性缩放规则来调整学习速率。 虽然这个指南可以在早期的工作[21,4]中找到，但它的经验限制并没有得到很好的理解，而且非正式地我们发现研究界并不广为人知。 为了成功应用这一规则，我们提出了一种新的预热策略，即在训练开始时使用较低学习率的策略[16]，以克服早期优化困难。 重要的是，我们的方法不仅与基线验证错误相匹配，而且还产生与小型小批量基线紧密匹配的训练误差曲线。 详情见§2。

Our comprehensive experiments in §5 show that optimization difficulty is the main issue with large minibatches, rather than poor generalization (at least on ImageNet), in contrast to some recent studies [20]. Additionally, we show that the linear scaling rule and warmup generalize to more complex tasks including object detection and instance segmentation[9, 31, 14, 28], which we demonstrate via the recently developed Mask R-CNN [14]. We note that a robust and successful guideline for addressing a wide range of minibatch sizes has not been presented in previous work.

我们在§5中进行的全面实验表明，与最近的一些研究相比，优化难度是大型微型计算机的主要问题，而不是较差的泛化（至少在ImageNet上）[20]。此外，我们展示线性缩放规则和预热推广到更复杂的任务，包括对象检测和实例分割[9,31,14,28]，我们通过最近开发的Mask R-CNN [14]证明了这一点。我们注意到，以前的工作尚未提出一个强有力且成功的解决各种小批量尺寸的指南。

While the strategy we deliver is simple, its successful application requires correct implementation with respect to seemingly minor and often not well understood implementation details within deep learning libraries. Subtleties in the implementation of SGD can lead to incorrect solutions that are difficult to discover. To provide more helpful guidance we describe common pitfalls and the relevant implementation details that can trigger these traps in §3.

虽然我们提供的策略很简单，但它的成功应用需要在深度学习库中对看似微不足道且通常不太了解的实现细节进行正确实施。实施新元的微妙之处可能导致难以发现的错误解决方案。为了提供更有用的指导，我们描述了常见的陷阱以及可能在§3中触发这些陷阱的相关实现细节。

Our strategy applies regardless of framework, but achieving efficient linear scaling requires nontrivial communication algorithms. We use the open-source Caffe21 deep learning framework and Big Basin GPU servers [24], which operates efficiently using standard Ethernet networking(as opposed to specialized network interfaces). We describe the systems algorithms that enable our approach to operate near its full potential in §4.

无论框架如何，我们的策略都适用，但实现有效的线性扩展需要非常重要的通信算法。 我们使用开源Caffe21深度学习框架和Big Basin GPU服务器[24]，它使用标准以太网网络（而不是专用网络接口）高效运行。 我们描述了系统算法，使我们的方法能够在§4中充分发挥其潜力。

The practical advances described in this report are helpful across a range of domains. In an industrial domain, our system unleashes the potential of training visual models from internet-scale data, enabling training with billions of images per day. Of equal importance, in a research domain, we have found it to simplify migrating algorithms from a single-GPU to a multi-GPU implementation without requiring hyper-parameter search, e.g. in our experience migrating Faster R-CNN [31] and ResNets [16] from 1 to 8 GPUs.

本报告中描述的实际进展有助于各个领域。 在工业领域，我们的系统释放了从互联网规模数据培训视觉模型的潜力，每天可以培训数十亿张图像。 同样重要的是，在研究领域，我们发现它可以简化将算法从单GPU迁移到多GPU实现而无需超参数搜索，例如： 根据我们的经验，将速度更快的R-CNN [31]和ResNets [16]从1个GPU迁移到8个GPU。

## 2,Large Minibatch SGD
We start by reviewing the formulation of Stochastic Gradient Descent (SGD), which will be the foundation of our discussions in the following sections. We consider supervised learning by minimizing a loss L(w) of the form:

我们首先回顾一下随机梯度下降（SGD）的表述，这将是我们在以下章节中讨论的基础。 我们通过最小化表格的损失L（w）来考虑监督学习：

**数学公式**

Here w are the weights of a network, X is a labeled training set, and l(x, w) is the loss computed from samples x ∈ X and their labels y. Typically l is the sum of a classification loss (e.g., cross-entropy) and a regularization loss on w. 

这里w是网络的权重，X是标记的训练集，l（x，w）是从样本x∈X及其标签y计算的损失。 通常，l是分类损失（例如，交叉熵）和w上的正则化损失之和。

Minibatch Stochastic Gradient Descent [32], usually referred to as simply as SGD in recent literature even though it operates on minibatches, performs the following update:

Minibatch Stochastic Gradient Descent [32]，在最近的文献中通常简称为SGD，即使它在微型计算机上运行，也执行以下更新：

**数学公式**

这里B是从X采样的小批量，n = | B | 是小批量大小，η是学习率，t是迭代指数。 请注意，在实践中我们使用动量SGD; 我们回到§3中对动量的讨论。
### 2.1. Learning Rates for Large Minibatches
Our goal is to use large minibatches in place of small minibatches while maintaining training and generalization accuracy. This is of particular interest in distributed learning, because it can allow us to scale to multiple workers using simple data parallelism without reducing the per-worker workload and without sacrificing model accuracy.

我们的目标是使用大型小型客舱代替小型小型客车，同时保持培训和推广的准确性。 这对分布式学习特别感兴趣，因为它可以允许我们使用简单的数据并行性扩展到多个工作者，而不会减少每个工作者的工作量并且不会牺牲模型的准确性。

As we will show in comprehensive experiments, we found that the following learning rate scaling rule is surprisingly effective for a broad range of minibatch sizes:正如我们将在综合实验中展示的那样，我们发现以下学习速率缩放规则对于各种各样的小批量尺寸都非常有效：

**Linear Scaling Rule: When the minibatch size is multiplied by k, multiply the learning rate by k.线性缩放规则：当小批量大小乘以k时，将学习速率乘以k。**

All other hyper-parameters (weight decay, etc.) are kept unchanged. As we will show in §5, the linear scaling rule can help us to not only match the accuracy between using small and large minibatches, but equally importantly, to largely match their training curves, which enables rapid debugging and comparison of experiments prior to convergence.

所有其他超参数（重量衰减等）保持不变。 正如我们将在§5中所示，线性缩放规则不仅可以帮助我们匹配使用小型和大型微型计算机之间的准确性，而且同样重要的是，它们可以在很大程度上匹配它们的训练曲线，从而能够在收敛之前快速调试和比较实验。

**Interpretation.** We present an informal discussion of the linear scaling rule and why it may be effective. Consider a network at iteration t with weights wt, and a sequence of k minibatches Bj for 0 ≤ j < k each of size n. We compare the effect of executing k SGD iterations with small minibatches Bj and learning rate η versus a single iteration with a large minibatch ∪jBj of size kn and learning rate ηˆ.

**解释**。我们提出了线性缩放规则的非正式讨论以及它可能有效的原因。 考虑具有权重wt的迭代t的网络，以及每个大小为n的0≤j<k的k个小批量Bj的序列。 我们比较了执行k SGD迭代与小型小批量Bj和学习率η相比，具有大小kn和学习率η的大型小批量∪jBj的单次迭代的效果。

**这里还有一堆**

**Discussion.** The above linear scaling rule was adopted by Krizhevsky [21], if not earlier. However, Krizhevsky reported a 1% increase of error when increasing the minibatch size from 128 to 1024, whereas we show how to maintain accuracy across a much broader regime of minibatch sizes. Chen et al. [5] presented a comparison of numerous distributed SGD variants, and although their work also employed the linear scaling rule, it did not establish a small minibatch baseline. Li [25] (§4.6) showed distributed ImageNet training with minibatches up to 5120 without a loss in accuracy after convergence. However, their work did not demonstrate a hyper-parameter search-free rule for adjusting the learning rate as a function of minibatch size, which is a central contribution of our work.

**讨论**。如果不是更早的话，Krizhevsky [21]采用了上述线性缩放规则。然而，Krizhevsky报告说，当将小批量大小从128增加到1024时，误差增加了1％，而我们展示了如何在更广泛的小批量大小范围内保持准确性。陈等人。 [5]介绍了许多分布式SGD变体的比较，尽管他们的工作也使用了线性缩放规则，但它没有建立一个小的微型基线。 Li [25]（§4.6）展示了分布式ImageNet培训，其中包含高达5120的微型计算机，并且在收敛后没有精度损失。然而，他们的工作没有证明一个超参数无搜索规则来调整学习率作为小批量大小的函数，这是我们工作的核心贡献。

In recent work, Bottou et al. [4] (§4.2) review theoretical tradeoffs of minibatching and show that with the linear scaling rule, solvers follow the same training curve as a function of number of examples seen, and suggest the learning rate should not exceed a maximum rate independent of minibatch size (which justifies warmup). Our work empirically tests these theories with unprecedented minibatch sizes.

在最近的工作中，Bottou等人。 [4]（§4.2）回顾了微型化的理论权衡，并表明，利用线性缩放规则，求解器遵循相同的训练曲线作为所见例子的函数，并建议学习率不应超过独立于小批量的最大速率大小（证明热身）。我们的工作通过前所未有的小批量尺寸来验证这些理论。
### 2.2. Warmup
As we discussed, for large minibatches (e.g., 8k) the linear scaling rule breaks down when the network is changing rapidly, which commonly occurs in early stages of training. We find that this issue can be alleviated by a properly designed warmup [16], namely, a strategy of using less aggressive learning rates at the start of training.

正如我们所讨论的，对于大型小型机（例如，8k），线性缩放规则在网络快速变化时发生故障，这通常发生在训练的早期阶段。 我们发现这个问题可以通过适当设计的预热来缓解[16]，即在训练开始时使用较低攻击性学习率的策略。

**Constant warmup.** The warmup strategy presented in [16] uses a low constant learning rate for the first few epochs of training. As we will show in §5, we have found constant warmup particularly helpful for prototyping object detection and segmentation methods [9, 31, 26, 14] that fine-tune pre-trained layers together with newly initialized layers.
In our ImageNet experiments with a large minibatch of size kn, we have tried to train with the low learning rate of η for the first 5 epochs and then return to the target learning rate of ηˆ = kη. However, given a large k, we find that this constant warmup is not sufficient to solve the optimization problem, and a transition out of the low learning rate warmup phase can cause the training error to spike. This leads us to propose the following gradual warmup.

**不断的热身。** [16]中提出的预热策略在前几个训练时期使用低恒定学习率。 正如我们将在§5中展示的那样，我们发现恒定的预热特别有助于原型对象检测和分割方法[9,31,26,14]，它们将预先训练的层与新初始化的层一起微调。
在我们使用大小为kn的大型小批量的ImageNet实验中，我们尝试在前5个时期以低学习率η进行训练，然后返回到目标学习速率η=kη。 然而，给定大k，我们发现这种恒定的预热不足以解决优化问题，并且从低学习速率预热阶段的转换可能导致训练误差尖峰。 这导致我们提出以下渐进的热身。

**Gradual warmup.** We present an alternative warmup that gradually ramps up the learning rate from a small to a large value. This ramp avoids a sudden increase of the learning rate, allowing healthy convergence at the start of training. In practice, with a large minibatch of size kn, we start from a learning rate of η and increment it by a constant amount at each iteration such that it reaches ηˆ = kη after 5 epochs (results are robust to the exact duration of warmup). After the warmup, we go back to the original learning rate schedule.

逐渐热身。 我们提出了另一种预热方法，逐渐提高学习率，从小到大。 该斜坡避免了学习率的突然增加，从而在训练开始时实现健康的收敛。 在实践中，对于大小为kn的大型小批量，我们从学习速率η开始并在每次迭代时将其增加一个恒定量，使得它在5个时期之后达到η=kη（结果对于预热的确切持续时间是稳健的）。 在热身之后，我们回到原来的学习率计划。
### 2.3. Batch Normalization with Large Minibatches
Batch Normalization (BN) [19] computes statistics along the minibatch dimension: this breaks the independence of each sample’s loss, and changes in minibatch size change the underlying definition of the loss function being optimized. In the following we will show that a commonly used ‘shortcut’, which may appear to be a practical consideration to avoid communication overhead, is actually necessary for preserving the loss function when changing minibatch size.

批量标准化（BN）[19]计算沿着小批量维度的统计数据：这打破了每个样本损失的独立性，并且小批量大小的变化改变了被优化的损失函数的基础定义。 在下文中，我们将展示一个常用的“快捷方式”，这似乎是避免通信开销的实际考虑因素，实际上在更改小批量大小时保留损失函数是必要的。

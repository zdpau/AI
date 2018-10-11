Accurate, large minibatch SGD: training Imagenet in 1 hour

https://arxiv.org/pdf/1706.02677.pdf

## Abstract
Deep learning thrives with large neural networks and large datasets. However, larger networks and larger datasets result in longer training times that impede research and development progress. Distributed synchronous SGD offers a potential solution to this problem by dividing SGD minibatches over a pool of parallel workers. Yet to make this scheme efficient, the per-worker workload must be large, which implies nontrivial growth in the SGD minibatch size. In this paper, we empirically show that on the ImageNet dataset large minibatches cause optimization difficulties, but when these are addressed the trained networks exhibit good generalization. Specifically, we show no loss of accuracy when training with large minibatch sizes up to 8192 images. To achieve this result, we adopt a hyperparameter-free linear scaling rule for adjusting learning rates as a function of minibatch size and develop a new warmup scheme that overcomes optimization challenges early in training. With these simple techniques, our Caffe2-based system trains ResNet-50 with a minibatch size of 8192 on 256 GPUs in one hour, while matching small minibatch accuracy. Using commodity hardware, our implementation achieves ∼90% scaling efficiency when moving from 8 to 256 GPUs. Our findings enable training visual recognition models on internet-scale data with high efficiency.

深度学习通过大型神经网络和大型数据集而蓬勃发展。但是，较大的网络和较大的数据集会导致较长的培训时间，从而阻碍研究和开发进度。分布式同步SGD通过将SGD小批量分配给并行工作池来提供潜在的解决方案。然而，为了使该方案有效，每个工人的工作量必须很大，这意味着SGD小批量规模的重大增长。在本文中，我们凭经验表明，在ImageNet数据集上，large minibatches会导致优化困难，但是当这些问题得到解决时，经过训练的网络会表现出良好的generalization。具体而言，我们在使用large minibatch sizes up to 8192 images培训时，不会显示精度损失。为了实现这一结果，我们采用了一种超参数线性缩放规则，用于根据小批量大小调整学习率，并开发一种新的预热方案，以便在培训早期克服优化挑战。通过这些简单的技术，我们基于Caffe2的系统可以在一小时内对256个GPU上的minibatch size of 8192的ResNet-50进行训练，同时匹配小的小批量精度。使用商用硬件，我们的实现在从8个GPU移动到256个GPU时实现了约90％的扩展效率。我们的研究结果使得能够高效地在互联网规模数据上培训视觉识别模型。

## 1,Introduction
Scale matters. We are in an unprecedented era in AI research history in which the increasing data and model scale is rapidly improving accuracy in computer vision[22, 41, 34, 35, 36, 16], speech [17, 40], and natural language processing [7, 38]. Take the profound impact in computer vision as an example: visual representations learned by deep convolutional neural networks [23, 22] show excellent performance on previously challenging tasks like ImageNet classification [33] and can be transferred to difficult perception problems such as object detection and segmentation [8, 10, 28]. Moreover, this pattern generalizes: larger datasets and neural network architectures consistently yield improved accuracy across all tasks that benefit from pretraining[22, 41, 34, 35, 36, 16]. But as model and data scale grow, so does training time; discovering the potential and limits of large-scale deep learning requires developing novel techniques to keep training time manageable.

规模很重要。 我们正处于人工智能研究历史上前所未有的时代，其中不断增长的数据和模型规模正在迅速提高计算机视觉的准确性[22,41,34,35,36,16]，语音[17,40]和自然语言处理 [7,38]。 以计算机视觉中的深远影响为例：深度卷积神经网络[23,22]所学的视觉表征在以前具有挑战性的任务中表现出色，如ImageNet分类[33]，并且可以转移到难以理解的问题，如物体检测和分割[8,10,28]。 此外，这种模式概括：较大的数据集和神经网络架构始终在所有受益于预训练的任务中提高准确性[22,41,34,35,36,16]。但随着模型和数据规模的增长，培训时间也在增长; 发现大规模深度学习的潜力和局限需要开发新技术以保持训练时间的可控性。


The goal of this report is to demonstrate the feasibility of, and to communicate a practical guide to, large-scale training with distributed synchronous stochastic gradient descent(SGD). As an example, we scale ResNet-50 [16] training, originally performed with a minibatch size of 256 images(using 8 Tesla P100 GPUs, training time is 29 hours), to larger minibatches (see Figure 1). In particular, we show that with a large minibatch size of 8192, we can train ResNet-50 in 1 hour using 256 GPUs while maintaining the same level of accuracy as the 256 minibatch baseline.
While distributed synchronous SGD is now commonplace, no existing results show that generalization accuracy can be maintained with minibatches as large as 8192 or that such high-accuracy models can be trained in such short time.

本报告的目的是证明分布式同步随机梯度下降（SGD）的大规模训练的可行性，并传达实用指南。作为一个例子，我们扩展ResNet-50 [16]训练，最初使用256个图像的小批量（使用8个特斯拉P100 GPU，训练时间为29小时）到更大的minibatches（见图1）。特别是，我们表明，对于8192的大型小批量，我们可以使用256个GPU在1小时内训练ResNet-50，同时保持与256个小批量基线相同的准确度。
虽然分布式同步SGD现在司空见惯，但现有结果并未显示minibatches=8192可以保持通用精度，或者可以在如此短的时间内训练这种高精度模型。

To tackle this unusually large minibatch size, we employ a simple and hyperparameter-free linear scaling rule to adjust the learning rate. While this guideline is found in earlier work [21, 4], its empirical limits are not well understood and informally we have found that it is not widely known to the research community. To successfully apply this rule, we present a new warmup strategy, i.e., a strategy of using lower learning rates at the start of training [16], to overcome early optimization difficulties. Importantly, not only does our approach match the baseline validation error, but also yields training error curves that closely match the small minibatch baseline. Details are presented in §2.

为了解决这种异常大的小批量大小问题，我们采用了一种简单且无超参数的线性缩放规则来调整学习速率。虽然这个指南可以在早期的工作[21,4]中找到，但它的经验限制(empirical limits)并没有得到很好的理解，而且非正式地我们发现研究界并不广为人知。为了成功应用这一规则，我们提出了一种新的预热warmup策略，即在训练开始时使用较低学习率的策略[16]，以克服早期优化困难。重要的是，我们的方法不仅与基线验证错误相匹配，而且还产生与小型小批量基线紧密匹配的训练误差曲线。 详情见§2。

Our comprehensive experiments in §5 show that optimization difficulty is the main issue with large minibatches, rather than poor generalization (at least on ImageNet), in contrast to some recent studies [20]. Additionally, we show that the linear scaling rule and warmup generalize to more complex tasks including object detection and instance segmentation[9, 31, 14, 28], which we demonstrate via the recently developed Mask R-CNN [14]. We note that a robust and successful guideline for addressing a wide range of minibatch sizes has not been presented in previous work.

我们在§5中进行的全面实验表明，与最近的一些研究相比，优化难度是large minibatches的主要问题，而不是较差的泛化（至少在ImageNet上）[20]。此外，我们展示线性缩放规则和预热推广到更复杂的任务，包括对象检测和实例分割[9,31,14,28]，我们通过最近开发的Mask R-CNN[14]证明了这一点。我们注意到，以前的工作尚未提出一个强有力且成功的解决各种minibatch sizes的指南。

While the strategy we deliver is simple, its successful application requires correct implementation with respect to seemingly minor and often not well understood implementation details within deep learning libraries. Subtleties in the implementation of SGD can lead to incorrect solutions that are difficult to discover. To provide more helpful guidance we describe common pitfalls and the relevant implementation details that can trigger these traps in §3.

虽然我们提供的策略很简单，但它的成功应用需要在深度学习库中对看似微不足道且通常不太了解的实现细节进行正确实施。实现ＳＧＤ的微妙之处可能导致难以发现的错误解决方案。为了提供更有用的指导，我们描述了常见的陷阱以及可能在§3中触发这些陷阱的相关实现细节。

Our strategy applies regardless of framework, but achieving efficient linear scaling requires nontrivial communication algorithms. We use the open-source Caffe21 deep learning framework and Big Basin GPU servers [24], which operates efficiently using standard Ethernet networking(as opposed to specialized network interfaces). We describe the systems algorithms that enable our approach to operate near its full potential in §4.

无论框架如何，我们的策略都适用，但实现有效的线性扩展需要非常重要的通信算法。我们使用开源Caffe21深度学习框架和Big Basin GPU服务器[24]，它使用标准以太网网络（而不是专用网络接口）高效运行。我们描述了系统算法，使我们的方法能够在§4中充分发挥其潜力。

The practical advances described in this report are helpful across a range of domains. In an industrial domain, our system unleashes the potential of training visual models from internet-scale data, enabling training with billions of images per day. Of equal importance, in a research domain, we have found it to simplify migrating algorithms from a single-GPU to a multi-GPU implementation without requiring hyper-parameter search, e.g. in our experience migrating Faster R-CNN [31] and ResNets [16] from 1 to 8 GPUs.

本报告中描述的实际进展有助于各个领域。在工业领域，我们的系统释放了从互联网规模数据培训视觉模型的潜力，每天可以培训数十亿张图像。同样重要的是，在研究领域，我们发现它可以简化将算法从单GPU迁移到多GPU实现而无需超参数搜索，例如：根据我们的经验，将速度更快的R-CNN [31]和ResNets [16]从1个GPU迁移到8个GPU。

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

我们的目标是使用large minibatches代替small minibatches，同时保持训练和泛化的准确性。这对分布式学习特别感兴趣，因为它可以允许我们使用简单的数据并行性扩展到多个工作者，而不会减少每个工作者的工作量并且不会牺牲模型的准确性。

As we will show in comprehensive experiments, we found that the following learning rate scaling rule is surprisingly effective for a broad range of minibatch sizes:正如我们将在综合实验中展示的那样，我们发现以下学习速率缩放规则对于各种各样的小批量尺寸都非常有效：

**Linear Scaling Rule: When the minibatch size is multiplied by k, multiply the learning rate by k.线性缩放规则：当小批量大小乘以k时，将学习速率乘以k。**

All other hyper-parameters (weight decay, etc.) are kept unchanged. As we will show in §5, the linear scaling rule can help us to not only match the accuracy between using small and large minibatches, but equally importantly, to largely match their training curves, which enables rapid debugging and comparison of experiments prior to convergence.

所有其他超参数（权重衰减等）保持不变。 正如我们将在§5中所示，线性缩放规则不仅可以帮助我们匹配使用小型和大型minibatches之间的准确性，而且同样重要的是，它们可以在很大程度上匹配它们的训练曲线，从而能够在收敛之前快速调试和比较实验。

**Interpretation.** We present an informal discussion of the linear scaling rule and why it may be effective. Consider a network at iteration t with weights wt, and a sequence of k minibatches Bj for 0 ≤ j < k each of size n. We compare the effect of executing k SGD iterations with small minibatches Bj and learning rate η versus a single iteration with a large minibatch ∪jBj of size kn and learning rate ηˆ.

**解释**。我们提出了线性缩放规则的非正式讨论以及它可能有效的原因。考虑具有权重wt的迭代t的网络，以及每个大小为n的0≤j<k的k个小批量Bj的序列。我们比较了执行k次SGD迭代与small minibatches　Bj和学习率η相比，具有大小kn和学习率η的大型小批量∪jBj的单次迭代的效果。

根据（2），在学习率η和小批量大小为n的SGD迭代之后，我们得到：**公式３**

另一方面，使用大小为kn的大型小批量∪jBj和学习率η进行一步产生：**公式4**

As expected, the updates differ, and it is unlikely that wˆt+1 = wt+k. However, if we could assume ∇l(x, wt) ≈ ∇l(x, wt+j ) for j < k, then setting ηˆ = kη would yield wˆt+1 ≈ wt+k, and the updates from small and large minibatch SGD would be similar. Although this is a strong assumption, we emphasize that if it were true the two updates are similar only if we set ηˆ = kη.

正如所料，更新不同，wt + 1 = wt + k不太可能。但是，如果我们可以假设j<k时，∇l（x，wt）≈∇l（x，wt + j），然后设置η=kη将产生wt +1≈wt+ k，并且来自小型和大型的minibatch SGD的更新将是类似的。虽然这是一个强有力的假设，但我们强调如果确实如此，那么只有当我们设置η=kη时，两个更新才是相似的。

The above interpretation gives intuition for one case where we may hope the linear scaling rule to apply. In our experiments with ηˆ = kη (and warmup), small and large minibatch SGD not only result in models with the same final accuracy, but also, the training curves match closely. Our empirical results suggest that the above approximation might be valid in large-scale, real-world data.

上述解释给出了一个我们可能希望应用线性缩放规则的情况的直觉。在我们使用η=kη（和预热）的实验中，小型和大型小批量SGD不仅导致具有相同最终精度的模型，而且训练曲线也紧密匹配。我们的实证结果表明，上述近似可能在大规模的实际数据中有效。

However, there are at least two cases when the condition ∇l(x, wt) ≈ ∇l(x, wt+j ) will clearly not hold. First, in initial training when the network is changing rapidly, it does not hold. We address this by using a warmup phase, discussed in §2.2. Second, minibatch size cannot be scaled indefinitely: while results are stable for a large range of sizes, beyond a certain point accuracy degrades rapidly. Interestingly, this point is as large as ∼8k in ImageNet experiments.

然而，当条件∇l（x，wt）≈1（x，wt + j）显然不成立时，至少存在两种情况。首先，在网络快速变化的初始培训中，它不成立。我们通过使用§2.2中讨论的预热阶段来解决这个问题。其次，小批量大小无法无限扩展：虽然结果对于大范围的大小是稳定的，但超过某一点时，准确度会迅速降低。有趣的是，这一点在ImageNet实验中大到了〜8k。

**Discussion.** The above linear scaling rule was adopted by Krizhevsky [21], if not earlier. However, Krizhevsky reported a 1% increase of error when increasing the minibatch size from 128 to 1024, whereas we show how to maintain accuracy across a much broader regime of minibatch sizes. Chen et al. [5] presented a comparison of numerous distributed SGD variants, and although their work also employed the linear scaling rule, it did not establish a small minibatch baseline. Li [25] (§4.6) showed distributed ImageNet training with minibatches up to 5120 without a loss in accuracy after convergence. However, their work did not demonstrate a hyper-parameter search-free rule for adjusting the learning rate as a function of minibatch size, which is a central contribution of our work.

**讨论**。如果不是更早的话，Krizhevsky[21]采用了上述线性缩放规则。然而，Krizhevsky报告说，当将小批量大小从128增加到1024时，误差增加了1％，而我们展示了如何在更广泛的小批量大小范围内保持准确性。陈等人[5]介绍了许多分布式SGD变体的比较，尽管他们的工作也使用了线性缩放规则，但它没有建立一个小的minibatch基线。Li [25]（§4.6）展示了分布式ImageNet培训，其中包含高达5120的minibatches，并且在收敛后没有精度损失。然而，他们的工作没有证明一个超参数无搜索规则来调整学习率作为小批量大小的函数，这是我们工作的核心贡献。

In recent work, Bottou et al. [4] (§4.2) review theoretical tradeoffs of minibatching and show that with the linear scaling rule, solvers follow the same training curve as a function of number of examples seen, and suggest the learning rate should not exceed a maximum rate independent of minibatch size (which justifies warmup). Our work empirically tests these theories with unprecedented minibatch sizes.

在最近的工作中，Bottou等人[4]（§4.2）回顾了minibatching的理论权衡，并表明，利用线性缩放规则，求解器遵循相同的训练曲线作为所见例子的函数，并建议学习率不应超过独立于小批量的最大速率大小（证明热身）。我们的工作通过前所未有的小批量尺寸来验证这些理论。
### 2.2. Warmup
As we discussed, for large minibatches (e.g., 8k) the linear scaling rule breaks down when the network is changing rapidly, which commonly occurs in early stages of training. We find that this issue can be alleviated by a properly designed warmup [16], namely, a strategy of using less aggressive learning rates at the start of training.

正如我们所讨论的，对于大型minibatches（例如，8k），线性缩放规则在网络快速变化时发生故障，这通常发生在训练的早期阶段。我们发现这个问题可以通过适当设计的预热来缓解[16]，即在训练开始时使用较低攻击性学习率的策略。

**Constant warmup.** The warmup strategy presented in [16] uses a low constant learning rate for the first few epochs of training. As we will show in §5, we have found constant warmup particularly helpful for prototyping object detection and segmentation methods [9, 31, 26, 14] that fine-tune pre-trained layers together with newly initialized layers.
In our ImageNet experiments with a large minibatch of size kn, we have tried to train with the low learning rate of η for the first 5 epochs and then return to the target learning rate of ηˆ = kη. However, given a large k, we find that this constant warmup is not sufficient to solve the optimization problem, and a transition out of the low learning rate warmup phase can cause the training error to spike. This leads us to propose the following gradual warmup.

**不断的热身。** [16]中提出的预热策略在前几个训练时期使用低恒定学习率。正如我们将在§5中展示的那样，我们发现恒定的预热特别有助于原型对象检测和分割方法[9,31,26,14]，它们将预先训练的层与新初始化的层一起微调。
在我们使用大小为kn的大型小批量的ImageNet实验中，我们尝试在前5个时期以低学习率η进行训练，然后返回到目标学习速率η=kη。然而，给定一个大k，我们发现这种恒定的预热不足以解决优化问题，并且从低学习速率预热阶段的转换可能导致训练误差尖峰。这导致我们提出以下渐进的热身。

**Gradual warmup.** We present an alternative warmup that gradually ramps up the learning rate from a small to a large value. This ramp avoids a sudden increase of the learning rate, allowing healthy convergence at the start of training. In practice, with a large minibatch of size kn, we start from a learning rate of η and increment it by a constant amount at each iteration such that it reaches ηˆ = kη after 5 epochs (results are robust to the exact duration of warmup). After the warmup, we go back to the original learning rate schedule.

**逐渐热身**。 我们提出了另一种预热方法，逐渐提高学习率，从小到大。该斜坡避免了学习率的突然增加，从而在训练开始时实现健康的收敛。在实践中，对于大小为kn的大型minibatch，我们从学习速率η开始并在每次迭代时将其增加一个恒定量，使得它在5个时期之后达到η=kη（结果对于预热的确切持续时间是稳健的）。在warmup之后，我们回到原来的学习率计划。
### 2.3. Batch Normalization with Large Minibatches
Batch Normalization (BN) [19] computes statistics along the minibatch dimension: this breaks the independence of each sample’s loss, and changes in minibatch size change the underlying definition of the loss function being optimized. In the following we will show that a commonly used ‘shortcut’, which may appear to be a practical consideration to avoid communication overhead, is actually necessary for preserving the loss function when changing minibatch size.

批量标准化（BN）[19]计算沿着小批量维度的统计数据：这打破了每个样本损失的独立性，并且小批量大小的变化改变了被优化的损失函数的基础定义。在下文中，我们将展示一个常用的“快捷方式”，这似乎是避免通信开销的实际考虑因素，实际上在更改小批量大小时保留损失函数是必要的。

We note that (1) and (2) assume the per-sample loss l(x, w) is independent of all other samples. This is not the case when BN is performed and activations are computed across samples. We write lB(x, w) to denote that the loss of a single sample x depends on the statistics of all samples in its minibatch B. We denote the loss over a single minibatch B of size n as L(B, w) = 1/n见原文. With BN, the training set can be thought of as containing all distinct subsets of size n drawn from the original training set X, which we denote as Xn. The training loss L(w) then becomes:

我们注意到（1）和（2）假设每样本损失l（x，w）独立于所有其他样本。 当执行BN并且跨样本计算激活时，情况并非如此。我们写lB（x，w）来表示单个样本x的丢失取决于其小批量B中所有样本的统计数据。我们将大小为n的单个小批量B的损失表示为L（B，w）=。 对于BN，训练集可以被认为包含从原始训练集X中绘制的大小为n的所有不同子集，我们将其表示为Xn。然后训练损失L（w）变为：**公式５**

If we view B as a ‘single sample’ in Xn, then the loss of each single sample B is computed independently.如果我们将B视为Xn中的“单个样本”，则每个单个样本B的loss是独立计算的。

Note that the minibatch size n over which the BN statistics are computed is a key component of the loss: if the perworker minibatch sample size n is changed, it changes the underlying loss function L that is optimized. More specifically, the mean/variance statistics computed by BN with different n exhibit different levels of random variation.

注意，计算BN统计量的小批量大小n是损失的关键组成部分：如果改变了工作者小批量样本大小n，则它改变优化的基础损失函数L. 更具体地，由具有不同n的BN计算的均值/方差统计表现出不同水平的随机变化。

In the case of distributed (and multi-GPU) training, if the per-worker sample size n is kept fixed and the total minibatch size is kn, it can be viewed a minibatch of k samples with each sample Bj independently selected from Xn, so the underlying loss function is unchanged and is still defined in Xn. Under this point of view, in the BN setting after seeing k minibatches Bj , (3) and (4) become:

在分布式（和多GPU）训练的情况下，如果每个工人样本大小n保持固定并且总minibatch大小为kn，则可以查看k个样本的minibatch，其中每个样本Bj独立地从Xn中选择，所以潜在的损失函数没有变化，仍然在Xn中定义。在这种观点下，在看到k个小批量Bj之后的BN设置中，（3）和（4）变为：**公式６和７**

Following similar logic as in §2.1, we set ηˆ = kη and we keep the per-worker sample size n constant when we change the number of workers k.

遵循类似于§2.1的逻辑，我们设置η=kη，并且当我们改变工人数k时，我们保持每个工人的样本大小为n。

In this work, we use n = 32 which has performed well for a wide range of datasets and networks [19, 16]. If n is adjusted, it should be viewed as a hyper-parameter of BN, not of distributed training. We also note that the BN statistics should not be computed across all workers, not only for the sake of reducing communication, but also for maintaining the same underlying loss function being optimized.

在这项工作中，我们使用n = 32，它在各种数据集和网络中表现良好[19,16]。 如果调整n，则应将其视为BN的超参数，而不是分布式训练。 我们还注意到，不应该跨所有工作人员计算BN统计数据，这不仅是为了减少通信，而且是为了保持优化相同的基础损失函数。
## 3. Subtleties and Pitfalls of Distributed SGD
In practice a distributed implementation has many subtleties. Many common implementation errors change the definitions of hyper-parameters, leading to models that train but whose error may be higher than expected, and such issues can be difficult to discover. While the remarks below are straightforward, they are important to consider explicitly to faithfully implement the underlying solver.

在实践中，分布式实现具有许多细微之处。许多常见的实现错误改变了超参数的定义，导致训练的模型但其错误可能高于预期，并且这些问题可能难以发现。 虽然下面的评论很简单，但明确地考虑忠实地实现底层求解器是很重要的。

**Weight decay**Weight decay is actually the outcome of the gradient of an L2-regularization term in the loss function.More formally, the per-sample loss in (1) can be written as l(x, w) = λ/2llwll^2 + ε(x, w). Here λ/2llwll^2 is the sample-independent L2 regularization on the weights and ε(x, w) is a sample-dependent term such as the cross-entropy loss. The SGD update in (2) can be written as:

**Weight decay**权重衰减实际上是损失函数中L2正则项的梯度的结果。更正式地，（1）中的每样本损失可写为l（x，w）=λ/ 2llwll ^ 2 +ε（x，w）。 这里λ/ 2llwll ^ 2是权重上与样本无关的L2正则化，ε（x，w）是样本相关项，例如交叉熵损失。（2）中的SGD更新可写为：**这块完了好好看看** **公式８**

In practice, usually only the sample-dependent term ∑∇ε(x, wt) is computed by backprop; the term λwt is computed separately and added to the aggregated gradients contributed by ε(x, wt). If there is no weight decay term, there are many equivalent ways of scaling the learning rate, including scaling the term ε(x, wt). However, as can be seen from (8), in general this is not the case. We summarize these observations in the following remark:

在实践中，通常只有依赖于样本的项∑∇ε（x，wt）由backprop计算; 术语λwt是单独计算的，并加到由ε（x，wt）贡献的聚合梯度上。如果没有权重衰减项，则有许多等效的缩放学习率的方法，包括缩放术语ε（x，wt）。 但是，从（8）可以看出，一般情况并非如此。 我们在以下评论中总结了这些观察结果：

**Remark 1: Scaling the cross-entropy loss is not equivalent to scaling the learning rate.备注1：缩放交叉熵损失并不等于缩放学习率。**

Momentum correction. Momentum SGD is a commonly adopted modification to the vanilla SGD in (2). A reference implementation of momentum SGD has the following form:

动量修正。Momentum SGD是（2）中vanillaSGD的常用修饰。动量SGD的参考实现具有以下形式：**公式９**

Here m is the momentum decay factor and u is the update tensor. A popular variant absorbs the learning rate η into the update tensor. Substituting vt for ηut in (9) yields:

这里m是动量衰减因子，u是更新张量。 流行的变体将学习速率η吸收到更新张量中。 在（9）中用vt代替ηut得到：**公式１０**

For a fixed η, the two are equivalent. However, we note that while u only depends on the gradients and is independent of η, v is entangled with η. When η changes, to maintain equivalence with the reference variant in (9), the update for v should be:公式太麻烦了.We refer to the factor ηt+1/ηt as the momentum correction. We found that this is especially important for stabilizing training when ηt+1>>ηt, otherwise the history term vt is too small which leads to instability (for ηt+1 < ηt momentum correction is less critical). This leads to our second remark:

**Remark 2: Apply momentum correction after changing learning rate if using (10).**

对于固定的η，两者是等价的。 然而，我们注意到虽然u仅取决于梯度并且与η无关，但v与η纠缠在一起。 当η改变时，为了与（9）中的参考变量保持等价，v的更新应该是：公式。我们将因子ηt+ 1 /ηt称为动量校正。 我们发现，当ηt+ 1 >>ηt时，这对于稳定训练尤为重要，否则历史项vt太小而导致不稳定（对于ηt+ 1 <ηt动量校正不那么关键）。 这引出了我们的第二句话：

**备注2：如果使用（10），在改变学习率后应用动量校正。**

For k workers each with a perworker minibatch of size n, following (4), gradient aggregation must be performed over the entire set of kn examples according to 公式. Loss layers are typically implemented to compute an average loss over their local input, which amounts to computing a per-worker loss of l(x, wt)/n. Given this, correct aggregation requires averaging the k gradients in order to recover the missing 1/k factor. However, standard communication primitives like allreduce [11] perform summing, not averaging. Therefore, it is more efficient to absorb the 1/k scaling into the loss, in which case only the loss’s gradient with respect to its input needs to be scaled, removing the need to scale the entire gradient vector. We summarize this as follows:

对于每个具有大小为n的perworker小批量的k工人，在（4）之后，必须根据公式对整个kn示例集进行梯度聚合。 通常实现损失层以计算其本地输入的平均损失，这相当于计算每个工人的∑l（x，wt）/ n损失。 鉴于此，正确的聚合需要对k个梯度求平均，以便恢复丢失的1 / k因子。 然而，像allreduce [11]这样的标准通信原语执行求和，而不是求平均。 因此，将1 / k缩放吸收到损耗中更有效，在这种情况下，仅需要缩放相对于其输入的损耗梯度，从而无需缩放整个梯度向量。 我们总结如下：

**Remark 3: Normalize the per-worker loss by total minibatch size kn, not per-worker size n.备注3：将每个工人的损失标准化为总小批量大小kn，而非每个工人的大小n。**

We also note that it may be incorrect to ‘cancel k’ by setting ηˆ = η (not kη) and normalizing the loss by 1/n (not 1/kn), which can lead to incorrect weight decay (see Remark 1).

我们还注意到，通过设置η=η（不是kη）并将损失归一化1 / n（不是1 / kn）来“取消k”可能是不正确的，这可能导致不正确的权重衰减（参见备注1）。

**Data shuffling**SGD is typically analyzed as a process that samples data randomly with replacement. In practice, common SGD implementations apply random shuffling of the training set during each SGD epoch, which can give better results [3, 13]. To provide fair comparisons with baselines that use shuffling (e.g., [16]), we ensure the samples in one epoch done by k workers are from a single consistent random shuffling of the training set. To achieve this, for each epoch we use a random shuffling that is partitioned into k parts, each of which is processed by one of the k workers. Failing to correctly implement random shuffling in multiple workers may lead to noticeably different behavior, which may contaminate results and conclusions. In summary:

通常将SGD分析为随替换随机抽样数据的过程。 在实践中，常见的SGD实现在每个SGD时期期间应用训练集的随机改组，这可以给出更好的结果[3,13]。 为了提供与使用改组的基线的公平比较（例如，[16]），我们确保由k个工人完成的一个时期中的样本来自训练集的单个一致的随机改组。 为了达到这个目的，我们使用随机改组分为k个部分，每个部分由一个k工人处理。 未能在多个工作人员中正确实施随机改组可能会导致明显不同的行为，这可能会污染结果和结论。 综上所述：

**Remark 4: Use a single random shuffling of the training data (per epoch) that is divided amongst all k workers. 备注4：使用在所有k个工作者之间划分的训练数据（每个时期）的单个随机改组。**

## 4. Communication








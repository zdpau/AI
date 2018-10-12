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
In order to scale beyond the 8 GPUs in a single Big Basin server [24], gradient aggregation has to span across servers on a network. To allow for near perfect linear scaling, the aggregation must be performed in parallel with backprop. This is possible because there is no data dependency between gradients across layers. Therefore, as soon as the gradient for a layer is computed, it is aggregated across workers, while gradient computation for the next layer continues(as discussed in [5]). We give full details next.

为了扩展到单个Big Basin服务器中的8个GPU [24]，梯度聚合必须跨越网络上的服务器。 为了允许接近完美的线性缩放，聚合必须与backprop并行执行。 这是可能的，因为跨层的渐变之间没有数据依赖性。 因此，一旦计算了层的梯度，它就会在工人之间聚合，而下一层的梯度计算会继续（如[5]中所述）。 我们接下来详细介绍。
### 4.1. Gradient Aggregation
For every gradient, aggregation is done using an allreduce operation (similar to the MPI collective operation MPI Allreduce [11]). Before allreduce starts every GPU has its locally computed gradients and after allreduce completes every GPU has the sum of all k gradients. As the number of parameters grows and compute performance of GPUs increases, it becomes harder to hide the cost of aggregation in the backprop phase. Training techniques to overcome these effects are beyond the scope of this work (e.g., quantized gradients [18], Block-Momentum SGD [6]). However, at the scale of this work, collective communication was not a bottleneck, as we were able to achieve near-linear SGD scaling by using an optimized allreduce implementation.

对于每个梯度，使用allreduce操作进行聚合（类似于MPI集体操作MPI Allreduce [11]）。 在allreduce开始之前，每个GPU都有其本地计算的梯度，并且在allreduce完成之后，每个GPU都具有所有k个梯度的总和。 随着参数数量的增加和GPU的计算性能的提高，隐藏在backprop阶段的聚合成本变得更加困难。 克服这些影响的训练技术超出了这项工作的范围（例如，量化梯度[18]，Block-Momentum SGD [6]）。 然而，在这项工作的范围内，集体沟通不是瓶颈，因为我们能够通过使用优化的allreduce实现来实现接近线性的SGD缩放。

Our implementation of allreduce consists of three phases for communication within and across servers: (1) buffers from the 8 GPUs within a server are summed into a single buffer for each server, (2) the results buffers are shared and summed across all servers, and finally (3) the results are broadcast onto each GPU. For the local reduction and broadcast in phases (1) and (3) we used NVIDIA Collective Communication Library (NCCL)3 for buffers of size 256KB or more and a simple implementation consisting of a number of GPU-to-host memory copies and a CPU reduction otherwise. NCCL uses GPU kernels to accelerate intraserver collectives, so this approach dedicates more time on the GPU to backprop while using the CPU resources that would otherwise have been idle to improve throughput.

我们对allreduce的实现包括服务器内部和服务器之间通信的三个阶段：（1）服务器内8个GPU的缓冲区总和为每个服务器的单个缓冲区，（2）结果缓冲区在所有服务器之间共享和求和， 最后（3）将结果广播到每个GPU上。 对于分阶段（1）和（3）的本地缩减和广播，我们使用NVIDIA集体通信库（NCCL）3用于256KB或更大的缓冲区以及由许多GPU到主机内存副本和a组成的简单实现 否则CPU减少。 NCCL使用GPU内核来加速内部服务器集合，因此这种方法在GPU上花费更多时间来反向支持，同时使用原本闲置的CPU资源来提高吞吐量。

For interserver allreduce, we implemented two of the best algorithms for bandwidth-limited scenarios: the recursive halving and doubling algorithm [30, 37] and the bucket algorithm (also known as the ring algorithm) [2]. For both, each server sends and receives (2 p−1/p b) bytes of data, where b is the buffer size in bytes and p is the number of servers. While the halving/doubling algorithm consists of 2 log2(p) communication steps, the ring algorithm consists of 2(p − 1) steps. This generally makes the halving/doubling algorithm faster in latency-limited scenarios(i.e., for small buffer sizes and/or large server counts). In practice, we found the halving/doubling algorithm to perform much better than the ring algorithm for buffer sizes up to a million elements (and even higher on large server counts). On 32 servers (256 GPUs), using halving/doubling led to a speedup of 3× over the ring algorithm.

对于服务器间的allreduce，我们为带宽受限的情况实现了两种最佳算法：递归减半和加倍算法[30,37]和桶算法（也称为环算法）[2]。 对于两者，每个服务器发送和接收（2 p-1 / p b）个字节的数据，其中b是以字节为单位的缓冲区大小，p是服务器的数量。 虽然减半/加倍算法由2个log2（p）通信步骤组成，但环算法由2（p-1）个步骤组成。 这通常使延迟/加倍算法在延迟受限的情况下更快（即，对于小的缓冲区大小和/或大的服务器计数）。 在实践中，我们发现减半/加倍算法的性能比环算法要好得多，缓冲区大小可达一百万个元素（在大型服务器数量上甚至更高）。在32台服务器（256个GPU）上，使用减半/加倍导致环形算法的速度提高3倍。

The halving/doubling algorithm consists of a reducescatter collective followed by an allgather. In the first step of reduce-scatter, servers communicate in pairs (rank 0 with 1, 2 with 3, etc.), sending and receiving for different halves of their input buffers. For example, rank 0 sends the second half of its buffer to 1 and receives the first half of the buffer from 1. A reduction over the received data is performed before proceeding to the next step, where the distance to the destination rank is doubled while the data sent and received is halved. After the reduce-scatter phase is finished, each server has a portion of the final reduced vector.

减半/加倍算法包括reducecatter集合，后跟allgather。在reduce-scatter的第一步中，服务器成对通信（0级，1,2级，3级等），发送和接收输入缓冲器的不同部分。例如，等级0将其缓冲区的后半部分发送到1并从1接收缓冲区的前半部分。在继续下一步骤之前执行接收数据的减少，其中到目的地等级的距离加倍，而发送和接收的数据减半。在减少分散阶段完成之后，每个服务器具有最终减少的向量的一部分。


This is followed by the allgather phase, which retraces the communication pattern from the reduce-scatter in reverse, this time simply concatenating portions of the final reduced vector. At each server, the portion of the buffer that was being sent in the reduce-scatter is received in the allgather, and the portion that was being received is now sent.

接下来是allgather阶段，其反向地从reduce-scatter回溯通信模式，这次简单地连接最终缩减矢量的部分。在每个服务器上，在全部收集中接收在reduce-scatter中发送的缓冲区部分，现在发送正在接收的部分。

To support non-power-of-two number of servers, we used the binary blocks algorithm [30]. This is a generalized version of the halving/doubling algorithm where servers are partitioned into power-of-two blocks and two additional communication steps are used, one immediately after the intrablock reduce-scatter and one before the intrablock allgather. Non-power-of-two cases have some degree of load imbalance compared to power-of-two, though in our runs we did not see significant performance degradation.

为了支持非二次幂的服务器，我们使用了二进制块算法[30]。这是对分/加倍算法的通用版本，其中服务器被划分为两个幂块，并且使用两个额外的通信步骤，一个在intrablock reduce-scatter之后，一个在intrablock allgather之前。与二次幂相比，非二次幂的情况具有一定程度的负载不平衡，但在我们的运行中，我们没有看到显着的性能下降。
### 4.2. Software
The allreduce algorithms described are implemented in Gloo, a library for collective communication. It supports multiple communication contexts, which means no additional synchronization is needed to execute multiple allreduce instances in parallel. Local reduction and broadcast(described as phases (1) and (3)) are pipelined with interserver allreduce where possible.

描述的allreduce算法在Gloo中实现，Gloo是一个用于集体通信的库。它支持多个通信上下文，这意味着不需要额外的同步来并行执行多个allreduce实例。局部缩减和广播（描述为阶段（1）和（3））在可能的情况下使用交叉点allreduce进行流水线操作。

Caffe2 supports multi-threaded execution of the compute graph that represents a training iteration. Whenever there is no data dependency between subgraphs, multiple threads can execute those subgraphs in parallel. Applying this to backprop, local gradients can be computed in sequence, without dealing with allreduce or weight updates. This means that during backprop, the set of runnable subgraphs may grow faster than we can execute them. For subgraphs that contain an allreduce run, all servers must choose to execute the same subgraph from the set of runnable subgraphs. Otherwise, we risk distributed deadlock where servers are attempting to execute non-intersecting sets of subgraphs. With allreduce being a collective operation, servers would time out waiting. To ensure correct execution we impose a partial order on these subgraphs. This is implemented using a cyclical control input, where completion of the n-th allreduce unblocks execution of the (n + c)-th allreduce, with c being the maximum number of concurrent allreduce runs. Note that this number should be chosen to be lower than the number of threads used to execute the full compute graph.

Caffe2支持表示训练迭代的计算图的多线程执行。只要子图之间没有数据依赖关系，多个线程就可以并行执行这些子图。将此应用于backprop，可以按顺序计算局部梯度，而无需处理allreduce或权重更新。这意味着在backprop期间，可运行子图集的增长速度可能比我们执行它们的速度快。对于包含allreduce运行的子图，所有服务器必须选择从runnable子图集中执行相同的子图。否则，我们冒着分布式死锁的风险，其中服务器试图执行非交叉的子图集。由于allreduce是一个集体操作，服务器会超时等待。为确保正确执行，我们对这些子图施加了部分订单。这是使用循环控制输入实现的，其中第n个allreduce的完成解除阻塞第（n + c）个allreduce的执行，其中c是并发allreduce运行的最大数量。请注意，此数字应选择为低于用于执行完整计算图形的线程数。

### 4.3. Hardware
We used Facebook’s Big Basin [24] GPU servers for our experiments. Each server contains 8 NVIDIA Tesla P100 GPUs that are interconnected with NVIDIA NVLink. For local storage, each server has 3.2TB of NVMe SSDs. For network connectivity, the servers have a Mellanox ConnectX-4 50Gbit Ethernet network card and are connected to Wedge100 [1] Ethernet switches.

我们使用Facebook的Big Basin [24] GPU服务器进行实验。每台服务器包含8个与NVIDIA NVLink互连的NVIDIA Tesla P100 GPU。对于本地存储，每台服务器都有3.2TB的NVMe SSD。对于网络连接，服务器具有Mellanox ConnectX-4 50Gbit以太网网卡，并连接到Wedge100 [1]以太网交换机。

We have found 50Gbit of network bandwidth sufficient for distributed synchronous SGD for ResNet-50, per the following analysis. ResNet-50 has approximately 25 million parameters. This means the total size of parameters is 25 · 10^6 · sizeof(float) = 100MB. Backprop for ResNet-50 on a single NVIDIA Tesla P100 GPU takes 120 ms. Given that allreduce requires ∼2× bytes on the network compared to the value it operates on, this leads to a peak bandwidth requirement of 200MB/0.125s = 1600MB/s, or 12.8 Gbit/s, not taking into account communication overhead. When we add a smudge factor for network overhead, we reach a peak bandwidth requirement for ResNet-50 of ∼15 Gbit/s. 
As this peak bandwidth requirement only holds during backprop, the network is free to be used for different tasks that are less latency sensitive then aggregation (e.g. reading data or saving network snapshots) during the forward pass.

根据以下分析，我们已经发现50Gbit的网络带宽足以用于ResNet-50的分布式同步SGD。ResNet-50有大约2500万个参数。这意味着参数的总大小为25·10^6·sizeof（float）= 100MB。在单个NVIDIA Tesla P100 GPU上用于ResNet-50的Backprop需要120 ms。鉴于allreduce在网络上需要~2×字节与其运行的值相比，这导致峰值带宽要求为200MB/0.125s = 1600MB/s，或12.8Gbit/s，而不考虑通信开销。当我们为网络开销添加污迹因子时，我们达到了~15 Gbit/s的ResNet-50的峰值带宽要求。由于此峰值带宽要求仅在反向提升期间保持，因此网络可以自由地用于在前向传递期间对聚合（例如，读取数据或保存网络快照）具有较小延迟敏感性的不同任务。

## 5,Main Results and Analysis
Our main result is that we can train ResNet-50 [16] on ImageNet [33] using 256 workers in one hour, while matching the accuracy of small minibatch training. Applying the linear scaling rule along with a warmup strategy allows us to seamlessly scale between small and large minibatches (up to 8k images) without tuning additional hyper-parameters or impacting accuracy. In the following subsections we: (1) describe experimental settings, (2) establish the effectiveness of large minibatch training, (3) perform a deeper experimental analysis, (4) show our findings generalize to object detection/segmentation, and (5) provide timings.

我们的主要结果是我们可以在一小时内使用256名工人在ImageNet [33]上训练ResNet-50 [16]，同时匹配小型小批量培训的准确性。 应用线性缩放规则以及预热策略，我们可以在小型和大型小型机之间无缝扩展（最多8k图像），而无需调整额外的超参数或影响精度。 在以下小节中我们：（1）描述实验设置，（2）建立大型迷你训练的有效性，（3）进行更深入的实验分析，（4）显示我们的研究结果推广到物体检测/分割，以及（5） 提供时间安排。
### 5.1. Experimental Settings
The 1000-way ImageNet classification task [33] serves as our main experimental benchmark. Models are trained on the ∼1.28 million training images and evaluated by top1 error on the 50,000 validation images. 

1000路ImageNet分类任务[33]是我们的主要实验基准。 模型在~228万个训练图像上进行训练，并通过50,000个验证图像上的top1误差进行评估。

We use the ResNet-50 [16] variant from [12], noting that the stride-2 convolutions are on 3×3 layers instead of on 1×1 layers as in [16]. We use Nesterov momentum [29] with m of 0.9 following [12] but note that standard momentum as was used in [16] is equally effective. We use a weight decay λ of 0.0001 and following [16] we do not apply weight decay on the learnable BN coefficients (namely, γ and β in [19]). In order to keep the training objective fixed, which depends on the BN batch size n as described in §2.3, we use n = 32 throughout, regardless of the overall minibatch size. As in [12], we compute the BN statistics using running average (with momentum 0.9).

我们使用来自[12]的ResNet-50 [16]变体，注意到步幅-2卷绕在3×3层而不是在1×1层，如[16]。 我们使用Nesterov动量[29]，其中m为0.9 [12]，但请注意[16]中使用的标准动量同样有效。 我们使用0.0001的权重衰减λ，并且在[16]之后我们不对可学习的BN系数（即[19]中的γ和β）应用权重衰减。 为了保持培训目标的固定，这取决于§2.3中描述的BN批量大小n，我们始终使用n = 32，无论整体小批量大小如何。 如[12]所示，我们使用运行平均值（动量为0.9）计算BN统计量。

All models are trained for 90 epochs regardless of minibatch sizes. We apply the linear scaling rule from §2.1 and use a learning rate of η = 0.1 · (kn/256) that is linear in the minibatch size kn. With k = 8 workers (GPUs) and n = 32 samples per worker, η = 0.1 as in [16]. We call this number( 0.1 · (kn/256) ) the reference learning rate, and reduce it by 1/10 at the 30-th, 60-th, and 80-th epoch, similar to [16].

无论小批量大小如何，所有型号都经过90个时期的培训。我们应用第2.1节中的线性缩放规则，并使用在小批量大小kn中线性的学习率η= 0.1·（kn / 256）。当k = 8个工人（GPU）和每个工人n = 32个样本时，η= 0.1，如[16]中所示。我们将这个数字（0.1·（kn / 256））称为参考学习率，并在第30,60和80周时将其减少1/10，类似于[16]。

We adopt the initialization of [15] for all convolutional layers. The 1000-way fully-connected layer is initialized by drawing weights from a zero-mean Gaussian with standard deviation of 0.01. We have found that although SGD with a small minibatch is not sensitive to initialization due to BN, this is not the case for a substantially large minibatch. Additionally we require an appropriate warmup strategy to avoid optimization difficulties in early training. 


我们对所有卷积层采用[15]的初始化。通过从标准偏差为0.01的零均值高斯绘制权重来初始化1000路全连接层。我们发现虽然带有小型小批量的SGD对BN的初始化不敏感，但对于大型小型小批量来说情况并非如此。此外，我们需要适当的预热策略，以避免早期训练中的优化困难。

For BN layers, the learnable scaling coefficient γ is initialized to be 1, except for each residual block’s last BN where γ is initialized to be 0. Setting γ = 0 in the last BN of each residual block causes the forward/backward signal initially to propagate through the identity shortcut of ResNets, which we found to ease optimization at the start of training. This initialization improves all models but is particularly helpful for large minibatch training as we will show.

对于BN层，可学习的缩放系数γ被初始化为1，除了每个残余块的最后一个BN，其中γ被初始化为0.在每个残余块的最后一个BN中设置γ= 0导致前向/后向信号最初为通过ResNets的身份快捷方式传播，我们发现在训练开始时可以简化优化。这种初始化改进了所有模型，但对我们将展示的大型小批量培训特别有用。

We use scale and aspect ratio data augmentation [36] as in [12]. The network input image is a 224×224 pixel random crop from an augmented image or its horizontal flip. The input image is normalized by the per-color mean and standard deviation, as in [12].

我们使用比例和宽高比数据增加[36]，如[12]。网络输入图像是来自增强图像或其水平翻转的224×224像素随机裁剪。输入图像通过每色平均值和标准偏差归一化，如[12]中所示。

**Handling random variation**. As models are subject to random variation in training, we compute a model’s error rate as the median error of the final 5 epochs. Moreover, we report the mean and standard deviation (std) of the error from 5 independent runs. This gives us more confidence in our results and also provides a measure of model stability. The random variation of ImageNet models has generally not been reported in previous work (largely due to resource limitations). We emphasize that ignoring random variation may cause unreliable conclusions, especially if results are from a single trial, or the best of many.

**处理随机变化**。由于模型在训练中受到随机变化的影响，我们将模型的误差率计算为最后5个时期的中值误差。此外，我们报告了5次独立运行的误差的平均值和标准差（std）。这使我们对结果更有信心，并且还提供了模型稳定性的度量。 ImageNet模型的随机变化在以前的工作中一般没有报道（主要是由于资源限制）。我们强调忽略随机变异可能会导致不可靠的结论，特别是如果结果来自单个试验，或者最好的结果。

**Baseline**. Under these settings, we establish a ResNet-50 baseline using k = 8 (8 GPUs in one server) and n = 32 images per worker (minibatch size of kn = 256), as in [16]. Our baseline has a top-1 validation error of 23.60% ±0.12. As a reference, ResNet-50 from fb.resnet.torch [12] has 24.01% error, and that of the original ResNet paper [16] has 24.7% under weaker data augmentation.

**基线**。在这些设置下，我们使用k = 8（一台服务器中的8个GPU）和每个工作者n = 32个图像（kn = 256的小批量大小）建立ResNet-50基线，如[16]中所述。我们的基线具有23.60％±0.12的前1验证误差。作为参考，来自fb.resnet.torch [12]的ResNet-50有24.01％的错误，而最初的ResNet论文[16]的数据增加较弱，有24.7％。

### 5.2. Optimization or Generalization Issues?
We establish our main results on large minibatch training by exploring optimization and generalization behaviors. We will demonstrate that with a proper warmup strategy, large minibatch SGD can both match the training curves of small minibatch SGD and also match the validation error. In other words, in our experiments both optimization and generalization of large minibatch training matches that of small minibatch training. Moreover, in §5.4 we will show that these models exhibit good generalization behavior to the object detection/segmentation transfer tasks, matching the transfer quality of small minibatch models.

我们通过探索优化和泛化行为，在大型小批量培训上建立我们的主要成果。我们将证明，通过适当的预热策略，大型小批量SGD既可以匹配小批量SGD的训练曲线，也可以匹配验证错误。换句话说，在我们的实验中，大型小批量培训的优化和概括与小型小批量培训相匹配。此外，在§5.4中，我们将展示这些模型对物体检测/分割传递任务表现出良好的泛化行为，与小型小批量模型的传递质量相匹配。

For the following results, we use k = 256 and n = 32, which results in a minibatch size kn = 8k (we use ‘1k’ to denote 1024). As discussed, our baseline has a minibatch size of kn = 256 and a reference learning rate of η = 0.1. Applying the linear scaling rule gives η = 3.2 as the reference learning rate for our large minibatch runs. We test three warmup strategies as discussed in §2.2: no warmup, constant warmup with η = 0.1 for 5 epochs, and gradual warmup which starts with η = 0.1 and is linearly increased to η = 3.2 over 5 epochs. All models are trained from scratch and all other hyper-parameters are kept fixed. We emphasize that while better results for any particular minibatch size could be obtained by optimizing hyper-parameters for that case; our goal is to match errors across minibatch sizes by using a general strategy that avoids hyper-parameter tuning for each minibatch size.

对于以下结果，我们使用k = 256和n = 32，这导致小批量大小kn = 8k（我们使用'1k'来表示1024）。如上所述，我们的基线的小批量大小为kn = 256，参考学习率为η= 0.1。应用线性缩放规则得到η= 3.2作为我们的大型小批量运行的参考学习率。我们测试了§2.2中讨论的三种预热策略：没有预热，恒定预热，η= 0.1，对于5个时期，逐渐预热，以η= 0.1开始，并在5个时期内线性增加到η= 3.2。所有模型都从头开始训练，所有其他超参数都保持固定。我们强调，通过优化该案例的超参数，可以获得任何特定小批量大小的更好结果;我们的目标是通过使用避免每个小批量大小的超参数调整的一般策略来匹配小批量大小的错误。

**Training error**. Training curves are shown in Figure 2. With no warmup (2a), the training curve for large minibatch of kn = 8k is inferior to training with a small minibatch of kn = 256 across all epochs. A constant warmup strategy(2b) actually degrades results: although the small constant learning rate can decrease error during warmup, the error spikes immediately after and training never fully recovers.

**训练错误**。训练曲线如图2所示。在没有预热（2a）的情况下，kn = 8k的大型小批量训练曲线不及在所有时期内使用kn = 256的小型小批量训练。恒定的预热策略（2b）实际上会降低结果：虽然小的恒定学习速率可以减少预热期间的错误，但是错误会在之后立即出现，并且训练永远不会完全恢复。

Our main result is that with gradual warmup, large minibatch training error matches the baseline training curve obtained with small minibatches, see Figure 2c. Although the large minibatch curve starts higher due to the low η in the warmup phase, it catches up shortly thereafter. After about 20 epochs, the small and large minibatch training curves match closely. The comparison between no warmup and gradual warmup suggests that large minibatch sizes are challenged by optimization difficulties in early training and if these difficulties are addressed, the training error and its curve can match a small minibatch baseline closely. 

我们的主要结果是，随着逐渐升温，大型小批量训练误差与小型小型飞机获得的基线训练曲线相匹配，见图2c。虽然由于预热阶段的η较低，大的小批量曲线开始较高，但此后不久就会赶上。大约20个时期后，小型和大型的迷你训练曲线紧密匹配。无预热和逐渐预热之间的比较表明，大型小批量大小受到早期训练中优化困难的挑战，如果解决了这些困难，训练误差及其曲线可以与小型小批量基线紧密匹配。

**Validation error**. Table 1 shows the validation error for the three warmup strategies. The no-warmup variant has ∼1.2% higher validation error than the baseline which is likely caused by the ∼2.1% increase in training error (Figure 2a), rather than overfitting or other causes for poor generalization.This argument is further supported by our gradual warmup experiment. The gradual warmup variant has a validation error within 0.14% of the baseline (noting that std of these estimates is ∼0.1%). Given that the final training errors (Figure 2c) match nicely in this case, it shows that if the optimization issues are addressed, there is no apparent generalization degradation observed using large minibatch training, even if the minibatch size goes from 256 to 8k.

**验证错误**。表1显示了三种预热策略的验证错误。无预热变量的验证误差比基线高约1.2％，这可能是由训练误差增加〜2.1％引起的（图2a），而不是过度拟合或其他原因造成的泛化不足。这一论点得到了我们的进一步支持。逐步热身实验。渐进式预热变量的验证误差在基线的0.14％范围内（注意这些估计值的标准值为〜0.1％）。鉴于最终训练错误（图2c）在这种情况下很好地匹配，它表明如果解决了优化问题，即使小批量大小从256到8k，使用大型小批量训练也没有观察到明显的泛化退化。

Finally, Figure 4 shows both the training and validation curves for the large minibatch training with gradual warmup. As can be seen, validation error starts to match the baseline closely after the second learning rate drop; actually, the validation curves can match earlier if BN statistics are recomputed prior to evaluating the error instead of using the running average (see also caption in Figure 4).

最后，图4显示了逐步预热的大型小批量训练的训练和验证曲线。可以看出，在第二次学习率下降后，验证错误开始与基线紧密匹配;实际上，如果在评估错误之前重新计算BN统计数据而不是使用运行平均值，则验证曲线可以更早匹配（参见图4中的标题）。

### 5.3. Analysis Experiments




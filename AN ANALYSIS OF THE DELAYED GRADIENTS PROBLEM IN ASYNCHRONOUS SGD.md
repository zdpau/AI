# AN ANALYSIS OF THE DELAYED GRADIENTS PROBLEM IN ASYNCHRONOUS SGD

## Abstract
### 梯度下降可以通过数据并行或模型并行在多个worker之间有效分配。 所有方法的共同目标是最大限度地减少工人的闲置时间。 参数同步（例如SGD中的每个小批次之后）需要参数服务器在应用更新之前等待最慢的工作人员回复。有一种完全异步的方法（Dean等，2012），最初称为Downpour SGD，它通过允许将陈旧参数上计算出的梯度发送到参数服务器来最小化工人空闲时间。 在实践中，异步SGD的直接使用会导致在从渐变梯度（during training from stale gradient）（称为“延迟梯度问题”）训练期间增加噪声，这会不准确地降低测试精度。延迟补偿，如Zheng等人详述的那样。 （2016a）以及各种暖启动计划可以帮助融合。在本文中，我们详细分析了在超参数选择的大范围扫描下由于延迟梯度引起的ASGD故障模式。 使用卷积模型，我们发现学习速率和批量选择是延迟梯度是否显著降低测试精度的主要因素。 仔细选择学习速率和批量大小，或使用自适应学习速率方法，可以有效地将延迟梯度问题减至最少（n = 257）。

## Introduction
### 神经网络训练已经通过模型并行模型（其中模型分散在不同的工作人员中）和数据并行性（其中训练数据被分解或分配给工作人员副本）扩展到许多工作人员。 分布式方法在模型参数如何同步方面有所不同，但最快（每单位时间计算的梯度）方法是完全异步的，并允许从陈旧参数计算梯度更新（Dean等，2012）。异步优化方法在共享集群中具有优势，其中单个工作人员可能会遇到来自并行作业，不同网络条件或异构硬件的缓慢下降。 陈等人（2017）添加额外的备份工作人员，以便运行同步SGD的参数服务器可以在最慢的工作人员回复之前继续下一批。
### 然而，由于梯度的延迟应用，ASGD遭受了准确度降低，为此已经提出了各种延迟补偿方法：
### •Dean等人。 （2012）发现Adagrad自适应学习率优化大大提高了Downpour SGD的稳健性。
### •陈等人。 （2017）发现，在前三个时期逐渐引入工人对于高延迟值下的稳定性非常重要。 作者还剪裁了梯度 for ASGD。
### •郑等人。 （2016b）将梯度函数的泰勒展开中的一阶项添加到延迟梯度提交。
### 为了增加同步训练中的每个worker的工作量，Goyal等人 （2017年）使用大型小批量（8k图像）分布在整个集群中，并且按比例大的学习率。 即使在ASGD中，mini-batches不在工人之间分配，大型小批量和学习率也会降低工人与服务器之间的通信频率。我们研究了在延迟梯度存在下，批量大小和学习速率对收敛的影响，以便更好地推荐ASGD的精确用例。

## method
### 在我们的实验中，我们训练了一个Lenet-5模型，用于MNIST数字识别任务超过30个时期（epochs），使用SGD(m = 0.9)。为了模拟异步SGD中的延迟梯度提交，我们创建了一个围绕同步PyTorch优化器的包装器，它在缓冲器中存储梯度并在恒定延迟之后再应用它们。
### 为了表明恒定时延同步SGD是完全异步SGD中忠实的梯度延迟模拟，我们对N个工作人员进行Monte-Carlo模拟，将N个梯度异步提交给参数服务器。假定worker处理和提交梯度的时间是正态分布的，（ that is, the time between gradient t and t0 arriving at the server from worker i is Ti,t0 − Ti,t ∼ N(1, σ2).）然后，我们发现梯度t’之前服务器上的更新次数达到了以N - 1为中心的正态分布（图1，其中σ= 0.2）。 尽管这一更新计数存在差异，但具有固定模拟延迟的单个worker近似ASGD使我们能够单独研究延迟梯度问题。 陈等人（2017）为了研究ASGD的行为，对一个worker进行类似的延迟梯度模拟。
### 最后，为了分析延迟梯度应用对测试精度的影响，我们在学习速率，批量大小和延迟的参数扫描上运行恒定延迟SGD。我们还将性能与Adam相比较，Adam预计对初始学习速率的敏感度较低。

## EXPERIMENTAL RESULTS
### 在不同提交延迟下模拟的异步SGD在低学习率（lr = 10-3）和低批量（b = 64，b = 128）下表现出最佳性能。 批量大于256且lr≥10-2.5时，测试精度显着降低。
### 我们观察到，无论延迟量，测试精度随LR平稳地向上变化，然后急剧下降。SGD时延越大，学习速率调整越不灵活，测试精度越快下降。在延迟的SGD中，在高学习率下测试准确度低的一个合理解释是，在训练开始时梯度变化更大，因此延迟更新会为参数搜索增加显着的噪音。的确，在Goyal等人 （2017年），逐步升温计划有效地用于减少同步小批量SGD的测试误差，具有较大的小批量大小和按比例缩放的学习速率。
### 然而，我们对Delayed SGD进行的实验表明，学习速率预热只会在初始时提高测试的准确性，但一旦应用完整学习速度后会立即下降。 学习率计划可能对减轻延迟梯度噪音影响不大。 我们发现使用每个参数自适应学习率的优化器（如Adam（Kingma＆Ba，2014））提高了学习速率和批量选择的弹性，与SGD无法处理超过32个延迟的时间相比，可实现与基线最高达256个延迟相当的测试精度。
### 最重要的是，我们观察到，即使对于many-worker场景（延迟= 32,128和256;图3），使用Adam手动调整的学习率和批量大小可以导致1-worker同步案例的结果相媲美（分别为98.21％，98.19％和97.94％）。我们用来实现这一目标的启发式算法大致是将学习率减半，并在workers数量增加一倍时将批量减半。 图（3）显示了这些情况下SGD和Adam之间测试准确性收敛的显着差异。

## 结论
### 我们的结果强调，学习率和批次大小是ASGD稳定性的主要因素。 此外，具有自适应优化器和仔细选择的批量大小的异步梯度方法可以成为快速模式的高效研究工具。

### 图1解释：模拟ASGD的更新延迟（分批）分布。正常配合以橙色叠加。
### 图2解释：同步和延迟更新方案下的MNIST测试准确性
### 图3解释：使用Adam代替SGD，即使对于许多批量大小的延迟梯度，LeNet收敛也可以在20个时期内达到。 即使如此，在更高的学习速度下，准确度仍然会大幅下降。 左边，观察到SGD在显着梯度延迟的情况下需要更长时间才能收敛。
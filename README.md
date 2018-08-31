# AI
## 有关ASGD、数据并行的讲解：https://blog.csdn.net/xbinworld/article/details/74781605
## 深度学习的各种概念：https://blog.csdn.net/bobpeter84/article/details/79136419
## Epoch, iteration, batch的理解：https://zhuanlan.zhihu.com/p/29409502 https://blog.csdn.net/program_developer/article/details/78597738
### 1，batch和batch size: 记住：batch size 和 number of batches 是不同的。(见上面专栏第二条换算关系)batchsize的正确选择是为了在内存效率和内存容量之间寻找最佳平衡。
既然有了mini batch那就会有一个batch size的超参数，也就是块大小。简单点说，batch size将决定**我们一次训练的样本数目**, batch_size将影响到模型的优化程度和速度。代表着每一个mini batch中有多少个样本。 我们一般设置为2的n次方。 例如64,128,512,1024. 一般不会超过这个范围。不能太大，因为太大了会无限接近full batch的行为，速度会慢。也不能太小，太小了以后可能算法永远不会收敛。
### 2, iteration:迭代是 batch 需要完成一个 epoch 的次数。在一个 epoch 中，batch 数和迭代数是相等的。
1个iteration等于使用batchsize个样本训练一次；
一个迭代 = 一个正向通过+一个反向通过
### 3, epoch:1个epoch等于使用训练集中的全部样本训练一次
一个epoch = 所有训练样本的一个正向传递和一个反向传递
训练集有1000个样本，batchsize=10，那么：训练完整个样本集需要：100次iteration，1次epoch。
## 吴恩达机器学习笔记
Two definitions of Machine Learning are offered. 
>* Arthur Samuel described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition.在不直接针对问题进行编程的情况下，赋予计算机学习能力的一个研究领域。
>* Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."对于某类任务T和性能度量P，如果计算机程序在T上以P衡量的性能随着经验E而自我完善，那么就称这个计算机程序从经验E学习。

Example: playing checkers.
>* E = the experience of playing many games of checkers
>* T = the task of playing checkers.
>* P = the probability that the program will win the next game.

In **supervised** learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

 Supervised learning problems are categorized into "regression" and "classification" problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.
 



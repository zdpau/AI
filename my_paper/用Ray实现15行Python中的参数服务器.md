https://ray-project.github.io/2018/07/15/parameter-server-in-fifteen-lines.html

Implementing A Parameter Server in 15 Lines of Python with Ray (用Ray实现15行Python中的参数服务器)

Parameter servers are a core part of many machine learning applications. Their role is to store the parameters of a machine learning model (e.g., the weights of a neural network) and to serve them to clients (clients are often workers that process data and compute updates to the parameters).

Parameter servers (like databases) are normally built and shipped as standalone systems. This post describes how to use Ray to implement a parameter server in a few lines of code.

参数服务器是许多机器学习应用程序的核心部分。它们的作用是存储机器学习模型的参数（例如，神经网络的权重），并为client提供服务（client通常是处理数据和计算参数更新的worker）。

参数服务器（如数据库）通常作为独立系统构建和交付。本文描述了如何使用Ray在几行代码中实现参数服务器。

By turning the parameter server from a “system” into an “application”, this approach makes it orders of magnitude simpler to deploy parameter server applications. Similarly, by allowing applications and libraries to implement their own parameter servers, this approach makes the behavior of the parameter server much more configurable and flexible (since the application can simply modify the implementation with a few lines of Python).

What is Ray? Ray is a general-purpose framework for parallel and distributed Python. Ray provides a unified task-parallel and actor abstraction and achieves high performance through shared memory, zero-copy serialization, and distributed scheduling. Ray also includes high-performance libraries targeting AI applications, for example hyperparameter tuning and reinforcement learning.

通过将参数服务器从“系统”转换为“应用程序”，这种方法使部署参数服务器应用程序更简单。同样，通过允许应用程序和库实现它们自己的参数服务器，这种方法使得参数服务器的行为更加可配置和灵活（因为应用程序只需使用几行Python就可以修改实现）。

什么是Ray？Ray是用于并行和分布式Python的通用框架。Ray提供统一的任务并行和actor抽象，并通过共享内存、零拷贝序列化和分布式调度实现高性能。Ray还包括针对人工智能应用的高性能库，例如超参数调整和强化学习。

## What is a Parameter Server?
A parameter server is a key-value store used for training machine learning models on a cluster. The **values** are the parameters of a machine-learning model (e.g., a neural network). The **keys** index the model parameters.

For example, in a movie recommendation system, there may be one key per user and one key per movie. For each user and movie, there are corresponding user-specific and movie-specific parameters. In a **language-modeling** application, words may act as keys and their embeddings may be the values. In its simplest form, a parameter server may implicitly have a single key and allow all of the parameters to be retrieved and updated at once. We show how such a parameter server can be implemented as a Ray actor (15 lines) below.

参数服务器是用于在集群上培训机器学习模型的key-value存储。这些**value**是机器学习模型（如神经网络）的参数。**key**索引模型参数。

例如，在电影推荐系统中，每个用户可能有一个密钥，每个电影有一个密钥。对于每个用户和电影，都有相应的用户特定参数和电影特定参数。在语言建模应用程序中，单词可能充当键，它们的嵌入可能是值。在最简单的形式中，参数服务器可以隐式地拥有一个键，并允许同时检索和更新所有参数。我们将在下面展示如何将这样一个参数服务器实现为一个Ray actor（15行）。
```
import numpy as np
import ray


@ray.remote
class ParameterServer(object):
    def __init__(self, dim):
        # Alternatively, params could be a dictionary mapping keys to arrays.
        # 或者，params可以是将键映射到数组的字典。
        self.params = np.zeros(dim)

    def get_params(self):
        return self.params

    def update_params(self, grad):
        self.params += grad
```
**The @ray.remote decorator defines a service**. It takes the ParameterServer class and allows it to be instantiated as a remote service or actor.

Here, we assume that the update is a gradient which should be added to the parameter vector. This is just the simplest possible example, and many different choices could be made.

**A parameter server typically exists as a remote process or service** and interacts with clients through remote procedure calls. To instantiate the parameter server as a remote actor, we can do the following.

**@ray.remote装饰器定义了一个service**。 它接受ParameterServer类并允许它实例化为remote service或actor。

这里，我们假设update是一个应该添加到参数向量的梯度。这只是最简单的例子，可以做出许多不同的选择。

**参数服务器通常作为远程进程或服务存在**，并通过远程过程调用与客户端进行交互。要将参数服务器实例化为远程actor，我们可以执行以下操作。
```
# We need to start Ray first.
ray.init()

# Create a parameter server process.
ps = ParameterServer.remote(10)
```
Actor method invocations return futures. If we want to retrieve the actual values, we can use a blocking ray.get call. For example, **Actor方法调用返回future**。如果我们想要检索实际值，我们可以使用阻塞ray.get调用。例如,
```
>>> params_id = ps.get_params.remote()  # This returns a future.

>>> params_id
ObjectID(7268cb8d345ef26632430df6f18cc9690eb6b300)

>>> ray.get(params_id)  # This blocks until the task finishes.
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
```
Now, suppose we want to start some worker tasks that continuously compute gradients and update the model parameters. Each worker will run in a loop that does three things: (现在，假设我们想要启动一些连续计算梯度并更新模型参数的worker任务。每个worker都将在一个循环中运行，该循环执行三件事)

1,Get the latest parameters. 获取最新参数。
2,Compute an update to the parameters. 计算参数的更新。
3,Update the parameters. 更新参数。
As a Ray remote function (though the worker could also be an actor), this looks like the following.
作为Ray远程函数（尽管worker也可以是actor），这看起来如下所示。
```
import time

# Note that the worker function takes a handle to the parameter server as an argument, which allows the worker task to invoke # methods on the parameter server actor. 请注意，worker函数将参数服务器的句柄作为参数，这允许worker任务调用参数服务器actor上的方法。

@ray.remote
def worker(ps):
    for _ in range(100):
        # Get the latest parameters.
        params_id = ps.get_params.remote()  # This method call is non-blocking and returns a future. 
                                            # 此方法调用是非阻塞的并返回Future
        params = ray.get(params_id)  # This is a blocking call which waits for the task to finish and gets the results.
                                     # 这是一个阻塞调用，等待任务完成并获得结果。

        # Compute a gradient update. Here we just make a fake update, but in practice this would use a library like TensorFlow # and would also take in a batch of data. 计算梯度更新。这里我们只是做一个虚假的更新，但实际上这将使用像TensorFlow这样的库，并且还会接收一批# # 数据。
        grad = np.ones(10)
        time.sleep(0.2)  # This is a fake placeholder for some computation.这是一个假的占位符，用于某些计算。

        # Update the parameters.
        ps.update_params.remote(grad)
```
Then we can start several worker tasks as follows.（然后我们可以按如下方式启动几个worker任务。）
```
# Start 2 workers.
for _ in range(2):
    worker.remote(ps)
```
Then we can retrieve the parameters from the driver process and see that they are being updated by the workers.
然后我们可以从driver进程中检索参数，并看到worker正在更新它们。

```
>>> ray.get(ps.get_params.remote())
array([64., 64., 64., 64., 64., 64., 64., 64., 64., 64.])
>>> ray.get(ps.get_params.remote())
array([78., 78., 78., 78., 78., 78., 78., 78., 78., 78.])
```
Part of the value that Ray adds here is that Ray makes it as easy to start up a remote service or actor as it is to define a Python class. Handles to the actor can be passed around to other actors and tasks to allow arbitrary and intuitive messaging and communication patterns. Current alternatives are much more involved. For example, consider how the equivalent runtime service creation and service handle passing would be done with GRPC.

Ray在这里添加的部分价值在于，Ray可以像创建Python类一样轻松启动远程服务或actor。对actor的句柄可以传递给其他actor和任务，以允许任意和直观的消息传递和通信模式。目前的替代方案涉及更多。例如，考虑如何使用GRPC完成等效的运行时服务创建和服务句柄传递(https://grpc.io/docs/tutorials/basic/python.html#defining-the-service)。

## Additional Extensions
Here we describe some important modifications to the above design. We describe additional natural extensions in this paper.

Sharding Across Multiple Parameter Servers: When your parameters are large and your cluster is large, a single parameter server may not suffice because the application could be bottlenecked by the network bandwidth into and out of the machine that the parameter server is on (especially if there are many workers).

A natural solution in this case is to shard the parameters across multiple parameter servers. This can be achieved by simply starting up multiple parameter server actors. An example of how to do this is shown in the code example at the bottom.

Controlling Actor Placement: The placement of specific actors and tasks on different machines can be specified by using Ray’s support for arbitrary resource requirements. For example, if the worker requires a GPU, then its remote decorator can be declared with @ray.remote(num_gpus=1). Arbitrary custom resources can be defined as well.

在这里，我们描述了对上述设计的一些重要修改。我们在本文中描述了其他自然扩展。

跨多个参数服务器进行分片：当参数很大且集群很大时，单个参数服务器可能不够，因为应用程序可能会受到进出参数服务器所在机器的网络带宽的限制（特别是如果有很多工作者）。

在这种情况下，一个自然的解决方案是在多个参数服务器上对参数进行分片。这可以通过简单地启动多个参数服务器actor来实现。底部的代码示例中显示了如何执行此操作的示例。

控制Actor放置：可以使用Ray对任意资源需求的支持来指定特定actor和任务在不同机器上的放置。例如，如果worker需要GPU，则可以使用@ ray.remote（num_gpus = 1）声明其远程装饰器。也可以定义任意的自定义资源。

## Unifying Tasks and Actors
Ray supports parameter server applications efficiently in large part due to its unified task-parallel and actor abstraction.

Popular data processing systems such as Apache Spark allow stateless tasks (functions with no side effects) to operate on immutable data. This assumption simplifies the overall system design and makes it easier for applications to reason about correctness.

However, mutable state that is shared between many tasks is a recurring theme in machine learning applications. That state could be the weights of a neural network, the state of a third-party simulator, or an encapsulation of an interaction with the physical world.

To support these kinds of applications, Ray introduces an actor abstraction. An actor will execute methods serially (so there are no concurrency issues), and each method can arbitrarily mutate the actor’s internal state. Methods can be invoked by other actors and tasks (and even by other applications on the same cluster).

One thing that makes Ray so powerful is that it unifies the actor abstraction with the task-parallel abstraction inheriting the benefits of both approaches. Ray uses an underlying dynamic task graph to implement both actors and stateless tasks in the same framework. As a consequence, these two abstractions are completely interoperable. Tasks and actors can be created from within other tasks and actors. Both return futures, which can be passed into other tasks or actor methods to introduce scheduling and data dependencies. As a result, Ray applications inherit the best features of both tasks and actors.

由于其统一的任务并行和actor抽象，Ray在很大程度上有效地支持参数服务器应用程序。

流行的数据处理系统（如Apache Spark）允许无状态任务（没有副作用的函数）对不可变数据进行操作。这种假设简化了整个系统的设计，使应用程序更容易推理正确性。

但是，在许多任务之间共享的可变状态是机器学习应用程序中反复出现的主题。该状态可以是神经网络的权重，第三方模拟器的状态，或者与物理世界的交互的封装。

为了支持这些类型的应用程序，Ray引入了一个actor抽象。 actor会串行执行方法（因此没有并发问题），每个方法都可以任意改变actor的内部状态。方法可以由其他actor和tasks（甚至是同一群集上的其他应用程序）调用。

让Ray如此强大的一件事是，它将actor抽象与任务并行抽象统一起来，继承了两种方法的优点。Ray使用底层动态任务图(DAG)在同一框架中实现actor和无状态任务。因此，这两个抽象是完全可互操作的。可以从其他task和actor中创建task和actor。两者都返回future，可以传递给其他任务或actor方法来引入调度和数据依赖。因此，Ray应用程序继承了任务和actor的最佳功能。

## Under the Hood
Dynamic Task Graphs: Under the hood, remote function invocations and actor method invocations create tasks that are added to a dynamically growing graph of tasks. The Ray backend is in charge of scheduling and executing these tasks across a cluster (or a single multi-core machine). Tasks can be created by the “driver” application or by other tasks.

Data: Ray efficiently serializes data using the Apache Arrow data layout. Objects are shared between workers and actors on the same machine through shared memory, which avoids the need for copies or deserialization. This optimization is absolutely critical for achieving good performance.

Scheduling: Ray uses a distributed scheduling approach. Each machine has its own scheduler, which manages the workers and actors on that machine. Tasks are submitted by applications and workers to the scheduler on the same machine. From there, they can be reassigned to other workers or passed to other local schedulers. This allows Ray to achieve substantially higher task throughput than what can be achieved with a centralized scheduler, which is important for machine learning applications.

动态任务图：在幕后，远程函数调用和actor方法调用创建添加到动态增长的任务图中的任务。Ray后端负责跨群集（或单个多核机器）安排和执行这些任务。任务可以由“driver”应用程序或其他任务创建。

数据：Ray使用Apache Arrow数据布局有效地序列化数据。通过共享内存在同一台机器上的工作者和演员之间共享对象，这避免了复制或反序列化的需要。这种优化对于实现良好性能至关重要。

调度：Ray使用分布式调度方法。每台机器都有自己的调度程序，用于管理该机器上的工作人员和演员。应用程序和工作人员将任务提交到同一台计算机上的调度程序。从那里，他们可以被重新分配给其他工作人员或传递给其他本地调度员。这使得Ray可以实现比集中式调度程序可以实现的任务吞吐量高得多的任务，这对于机器学习应用程序非常重要。
## conclusion
A parameter server is normally implemented and shipped as a standalone system. The thing that makes this approach so powerful is that we’re able to implement a parameter server with a few lines of code as an application. This approach makes it much simpler to deploy applications that use parameter servers and to modify the behavior of the parameter server. For example, if we want to shard the parameter server, change the update rule, switch between asynchronous and synchronous updates, ignore straggler workers, or any number of other customizations, we can do each of these things with a few extra lines of code.

This post describes how to use Ray actors to implement a parameter server. However, actors are a much more general concept and can be useful for many applications that involve stateful computation. Examples include logging, streaming, simulation, model serving, graph processing, and many others.

参数服务器通常作为独立系统实现和发布。使这种方法如此强大的原因是我们能够使用几行代码作为应用程序来实现参数服务器。这种方法使部署使用参数服务器的应用程序和修改参数服务器的行为变得更加简单。例如，如果我们想要对参数服务器进行分片，更改更新规则，在异步和同步更新之间切换，忽略straggler worker或任何其他自定义，我们可以使用一些额外的代码行来完成这些操作。

这篇文章描述了如何使用Ray actor实现参数服务器。但是，actor是一个更通用的概念，对于涉及有状态计算的许多应用程序都很有用。示例包括日志记录，流式传输，模拟，模型服务，图形处理等等。

##  implements a sharded parameter server
```
import numpy as np
import ray
import time

# Start Ray.
ray.init()


@ray.remote
class ParameterServer(object):
    def __init__(self, dim):
        # Alternatively, params could be a dictionary mapping keys to arrays.
        self.params = np.zeros(dim)

    def get_params(self):
        return self.params

    def update_params(self, grad):
        self.params += grad


@ray.remote
def worker(*parameter_servers):
    for _ in range(100):
        # Get the latest parameters.
        parameter_shards = ray.get(
          [ps.get_params.remote() for ps in parameter_servers])
        params = np.concatenate(parameter_shards)

        # Compute a gradient update. Here we just make a fake
        # update, but in practice this would use a library like
        # TensorFlow and would also take in a batch of data.
        grad = np.ones(10)
        time.sleep(0.2)  # This is a fake placeholder for some computation.
        grad_shards = np.split(grad, len(parameter_servers))

        # Send the gradient updates to the parameter servers.
        for ps, grad in zip(parameter_servers, grad_shards):
            ps.update_params.remote(grad)


# Start two parameter servers, each with half of the parameters.
parameter_servers = [ParameterServer.remote(5) for _ in range(2)]

# Start 2 workers.
workers = [worker.remote(*parameter_servers) for _ in range(2)]

# Inspect the parameters at regular intervals.定期检查参数
for _ in range(5):
    time.sleep(1)
    print(ray.get([ps.get_params.remote() for ps in parameter_servers]))
```

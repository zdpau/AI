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

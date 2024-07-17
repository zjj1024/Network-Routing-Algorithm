
# from _typeshed import Self
import torch
# from torch._C import FloatTensor
from torch.distributions import Categorical
from net import Actor, Critic
import numpy as np
import torch.nn as nn
# 智能体负责与环境交互并根据神经网络做出决策。
def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.1)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0)

class A2C:#net.py文件实现了他底层的两个函数，包括Actor和Critic两个网络
    def __init__(self,pa):
        # P初始化网络、优化器和一些参数。

        self.device = pa.device  # 获取设备信息（CPU或GPU）
        self.lr_actor = pa.lr_actor  # Actor网络的学习率
        self.lr_critic = pa.lr_critic  # Critic网络的学习率
        self.max_len = pa.max_ep_step  # 最大训练步数
        self.gamma = pa.gamma  # 折扣因子
        self.pa = pa  # 参数对象
        # SGD(model.parameters(), lr=0.1, momentum=0.9)
        # network initialization
        self.actor = Actor().apply(init_weights).to(self.device)  # 初始化Actor网络并应用权重初始化
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=pa.lr_actor)  # 初始化Actor网络的Adam优化器
        self.critic = Critic().apply(init_weights).to(self.device)  # 初始化Critic网络并应用权重初始化
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=pa.lr_critic)  # 初始化Critic网络的Adam优化器

        # for name, parameters in  self.actor.named_parameters():
        #     print(name, ':', parameters.size())

        total = sum([param.nelement() for param in self.actor.parameters()])
        print("Number of parameter: %.2fM" % (total/1e6))# 打印参数总数

        # 定义记忆
        self.memory_s = torch.zeros((1, self.pa.network_input_height, self.pa.network_input_length, self.pa.network_input_width))  # 初始化状态记忆
        self.memory_a = torch.zeros((1, self.pa.num_nw))  # 初始化动作记忆 num_nw=4
        self.memory_r = []  # 初始化奖励记忆
        self.counter = 0  # 初始化计数器

        self.idx = 0  # 初始化索引

    def save(self):
        torch.save(self.actor.state_dict(), 'actor_unb_4.pth.2.pth')  # 保存Actor网络的状态字典
        torch.save(self.critic.state_dict(), 'critic_unb4.pth.2.pth')  # 保存Critic网络的状态字典

    def load(self):
        self.actor.load_state_dict(torch.load('actor_unb_4.pth.2.pth'))  # 加载Actor网络的状态字典
        self.critic.load_state_dict(torch.load('critic_unb4.pth.2.pth'))  # 加载Critic网络的状态字典

    def test(self):
        self.actor.eval()  # 设置Actor网络为评估模式
        self.critic.eval()  # 设置Critic网络为评估模式

    def get_action(self, s):  # 定义获取动作的方法，参数s为当前状态

        self.idx += 1  # 索引值加1
        # 将状态转换为浮点张量，并在第0维增加一维，然后移动到指定设备（如GPU）
        s = torch.FloatTensor(s).unsqueeze(0).to(self.device)

        # print("unsq_s")
        # print(s)
        #
        # 打印actor (s)
        # print("actor (s)")
        # print(self.actor(s))
        # 通过actor网络获取动作的概率权重，并去除多余维度
        prob_weights = torch.squeeze(self.actor(s))

        # 打印prob_weights
        # print('prob_weights')
        # print(prob_weights)
        # tensor([[0.1240, 0.1252, 0.1249, 0.1257, 0.1252, 0.1252, 0.1250, 0.1248],
        #         [0.1260, 0.1241, 0.1264, 0.1237, 0.1241, 0.1273, 0.1242, 0.1243],
        #         [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250],
        #         [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]],
        #        grad_fn= < SqueezeBackward0 >)
        # 如果是训练阶段
        if self.pa.train == True:
            if self.idx % self.pa.various_epoch == 0:  # 每隔一定的epoch调整探索率
                # 调整最小探索率，但不低于0.0001
                self.pa.min_explore = max(self.pa.min_explore * self.pa.decay_min_explore, 0.0001)
                # print("打印self.pa.min_explore",self.pa.min_explore)
            # 将低于最小探索率的概率权重设置为最小探索率
            prob_weights[prob_weights < self.pa.min_explore] = self.pa.min_explore
            # print("打印prob_weights", prob_weights)

            # 根据动作的概率选择动作 # 通过多项式分布随机选择动作
            action = torch.multinomial(prob_weights, 1, replacement=False).detach()

        # 如果是测试阶段选择概率最大的动作
        else:
            action = torch.max(prob_weights, 1)[1]  # 选择概率最大的动作

        action = torch.squeeze(action)  # 去除多余维度
        # print("----------action----------",action)
        # ----------action - --------- tensor([5, 3, 1, 5])
        # ----------action - --------- tensor([4, 6, 4, 4])
        # ----------action - --------- tensor([0, 5, 2, 1])
        # ----------action - --------- tensor([0, 3, 0, 4])
        # ----------action - --------- tensor([6, 2, 0, 2])
        # ----------action - --------- tensor([2, 2, 4, 3])
        # ----------action - --------- tensor([0, 3, 3, 2])
        # ----------action - --------- tensor([2, 7, 7, 1])
        return action  # 返回选择的动作

    def learn(self, s, a, s_, r, done):
        """
        这个函数实现了智能体的学习过程，基于 Actor-Critic 方法进行策略优化。
        参数:
        s: 当前状态
        a: 执行动作
        s_: 新状态
        r: 奖励
        done: 是否终止
        """

        # 将状态 s 转换为张量，并增加一个维度
        s = torch.unsqueeze(torch.from_numpy(s), 0)
        # 将动作 a 转换为张量，并增加一个维度
        a = torch.unsqueeze(a, 0)

        # 如果是第一个样本，初始化记忆
        if self.counter == 0:
            self.memory_s = s
            self.memory_a = a
            self.memory_r.append(r)
        else:
            # 否则将新的样本添加到记忆中
            self.memory_s = torch.cat((self.memory_s, s), 0)
            self.memory_a = torch.cat((self.memory_a, a), 0)
            self.memory_r.append(r)

        # 计数器加一
        self.counter += 1

        # 如果计数器达到最大长度或结束标志为真，开始学习
        if self.counter >= self.max_len or done:
            # 计算折扣奖励
            discounted_r = []
            # 如果终止，则值为 0，否则通过 critic 预测新状态的值
            value_ = 0 if done else self.critic(torch.FloatTensor(s_).to(self.device)).item()
            # 反向遍历记忆中的奖励，计算折扣奖励
            for t in range(len(self.memory_r) - 1, -1, -1):
                value_ = value_ * self.gamma + self.memory_r[t]
                discounted_r.insert(0, value_)

            # 开始学习
            s = self.memory_s.type(torch.FloatTensor).to(self.device)
            a = self.memory_a.unsqueeze(-1).to(self.device)
            r = torch.FloatTensor(discounted_r).unsqueeze(-1).to(self.device)

            # 更新 critic
            v = self.critic(s)
            advantage = r - v
            critic_loss = torch.mean(torch.pow(advantage, 2))
            self.opt_critic.zero_grad()
            critic_loss.backward()
            # for param in self.critic.parameters():  # clip防止梯度爆炸
            #     param.grad.data.clamp_(-1, 1)
            self.opt_critic.step()

            # 更新 actor
            prob = self.actor(s)
            # 选择与动作概率对应的动作
            log_prob = torch.log(prob + 0.0001)
            log_prob = torch.gather(log_prob, 2, a).squeeze(-1)
            log_prob = (torch.sum(log_prob, 1).unsqueeze(-1)) / 4

            # 计算 actor 的损失
            actor_loss = -torch.sum(log_prob * advantage.detach())
            self.opt_actor.zero_grad()
            actor_loss.backward()
            # for param in self.actor.parameters():  # clip防止梯度爆炸
            #     param.grad.data.clamp_(-1, 1)
            self.opt_actor.step()

            # 重置记忆
            self.memory_s = torch.zeros((1, self.pa.network_input_height,
                                         self.pa.network_input_length, self.pa.network_input_width), dtype=float)
            self.memory_a = torch.zeros((1, self.pa.num_nw))
            self.memory_r = []
            self.counter = 0











# # from _typeshed import Self
# import torch
# # from torch._C import FloatTensor
# from torch.distributions import Categorical
# from net import Actor, Critic
# import numpy as np
# import torch.nn as nn
# # 智能体负责与环境交互并根据神经网络做出决策。
# def init_weights(layer):
#     # 如果为卷积层，使用正态分布初始化
#     if type(layer) == nn.Conv2d:
#         nn.init.normal_(layer.weight, mean=0, std=0.1)
#     # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
#     elif type(layer) == nn.Linear:
#         nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
#         nn.init.constant_(layer.bias, 0)

# class A2C:#net.py文件实现了他底层的两个函数，包括Actor和Critic两个网络
#     def __init__(self,pa):
#         # P初始化网络、优化器和一些参数。

#         self.device = pa.device  # 获取设备信息（CPU或GPU）
#         self.lr_actor = pa.lr_actor  # Actor网络的学习率
#         self.lr_critic = pa.lr_critic  # Critic网络的学习率
#         self.max_len = pa.max_ep_step  # 最大训练步数
#         self.gamma = pa.gamma  # 折扣因子
#         self.pa = pa  # 参数对象
#         # SGD(model.parameters(), lr=0.1, momentum=0.9)
#         # network initialization
#         self.actor = Actor().apply(init_weights).to(self.device)  # 初始化Actor网络并应用权重初始化
#         self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=pa.lr_actor)  # 初始化Actor网络的Adam优化器
#         self.critic = Critic().apply(init_weights).to(self.device)  # 初始化Critic网络并应用权重初始化
#         self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=pa.lr_critic)  # 初始化Critic网络的Adam优化器

#         # for name, parameters in  self.actor.named_parameters():
#         #     print(name, ':', parameters.size())

#         total = sum([param.nelement() for param in self.actor.parameters()])
#         print("Number of parameter: %.2fM" % (total/1e6))# 打印参数总数

#         # 定义记忆
#         self.memory_s = torch.zeros((1, self.pa.network_input_height, self.pa.network_input_length, self.pa.network_input_width))  # 初始化状态记忆
#         self.memory_a = torch.zeros((1, self.pa.num_nw))  # 初始化动作记忆 num_nw=4
#         self.memory_r = []  # 初始化奖励记忆
#         self.counter = 0  # 初始化计数器

#         self.idx = 0  # 初始化索引

#     def save(self):
#         torch.save(self.actor.state_dict(), 'actor_unb_4.pth.2.pth')  # 保存Actor网络的状态字典
#         torch.save(self.critic.state_dict(), 'critic_unb4.pth.2.pth')  # 保存Critic网络的状态字典

#     def load(self):
#         self.actor.load_state_dict(torch.load('actor_unb_4.pth.2.pth'))  # 加载Actor网络的状态字典
#         self.critic.load_state_dict(torch.load('critic_unb4.pth.2.pth'))  # 加载Critic网络的状态字典

#     def test(self):
#         self.actor.eval()  # 设置Actor网络为评估模式
#         self.critic.eval()  # 设置Critic网络为评估模式

#     def get_action(self, s):  # 定义获取动作的方法，参数s为当前状态

#         self.idx += 1  # 索引值加1
#         # 将状态转换为浮点张量，并在第0维增加一维，然后移动到指定设备（如GPU）
#         s = torch.FloatTensor(s).unsqueeze(0).to(self.device)

#         # print("unsq_s")
#         # print(s)
#         #
#         # 打印actor (s)
#         # print("actor (s)")
#         # print(self.actor(s))
#         # 通过actor网络获取动作的概率权重，并去除多余维度
#         prob_weights = torch.squeeze(self.actor(s))

#         # 打印prob_weights
#         # print('prob_weights')
#         # print(prob_weights)
#         # tensor([[0.1240, 0.1252, 0.1249, 0.1257, 0.1252, 0.1252, 0.1250, 0.1248],
#         #         [0.1260, 0.1241, 0.1264, 0.1237, 0.1241, 0.1273, 0.1242, 0.1243],
#         #         [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250],
#         #         [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]],
#         #        grad_fn= < SqueezeBackward0 >)
#         # 如果是训练阶段
#         if self.pa.train == True:
#             if self.idx % self.pa.various_epoch == 0:  # 每隔一定的epoch调整探索率
#                 # 调整最小探索率，但不低于0.0001
#                 self.pa.min_explore = max(self.pa.min_explore * self.pa.decay_min_explore, 0.0001)
#                 # print("打印self.pa.min_explore",self.pa.min_explore)
#             # 将低于最小探索率的概率权重设置为最小探索率
#             prob_weights[prob_weights < self.pa.min_explore] = self.pa.min_explore
#             # print("打印prob_weights", prob_weights)

#             # 根据动作的概率选择动作 # 通过多项式分布随机选择动作
#             action = torch.multinomial(prob_weights, 1, replacement=False).detach()

#         # 如果是测试阶段选择概率最大的动作
#         else:
#             action = torch.max(prob_weights, 1)[1]  # 选择概率最大的动作

#         action = torch.squeeze(action)  # 去除多余维度
#         # print("----------action----------",action)
#         # ----------action - --------- tensor([5, 3, 1, 5])
#         # ----------action - --------- tensor([4, 6, 4, 4])
#         # ----------action - --------- tensor([0, 5, 2, 1])
#         # ----------action - --------- tensor([0, 3, 0, 4])
#         # ----------action - --------- tensor([6, 2, 0, 2])
#         # ----------action - --------- tensor([2, 2, 4, 3])
#         # ----------action - --------- tensor([0, 3, 3, 2])
#         # ----------action - --------- tensor([2, 7, 7, 1])
#         return action  # 返回选择的动作

#     def learn(self, s, a, s_, r, done):
#         """
#         这个函数实现了智能体的学习过程，基于 Actor-Critic 方法进行策略优化。
#         参数:
#         s: 当前状态
#         a: 执行动作
#         s_: 新状态
#         r: 奖励
#         done: 是否终止
#         """

#         # 将状态 s 转换为张量，并增加一个维度
#         s = torch.unsqueeze(torch.from_numpy(s), 0)
#         # 将动作 a 转换为张量，并增加一个维度
#         a = torch.unsqueeze(a, 0)

#         # 如果是第一个样本，初始化记忆
#         if self.counter == 0:
#             self.memory_s = s
#             self.memory_a = a
#             self.memory_r.append(r)
#         else:
#             # 否则将新的样本添加到记忆中
#             self.memory_s = torch.cat((self.memory_s, s), 0)
#             self.memory_a = torch.cat((self.memory_a, a), 0)
#             self.memory_r.append(r)

#         # 计数器加一
#         self.counter += 1

#         # 如果计数器达到最大长度或结束标志为真，开始学习
#         if self.counter >= self.max_len or done:
#             # 计算折扣奖励
#             discounted_r = []
#             # 如果终止，则值为 0，否则通过 critic 预测新状态的值
#             value_ = 0 if done else self.critic(torch.FloatTensor(s_).to(self.device)).item()
#             # 反向遍历记忆中的奖励，计算折扣奖励
#             for t in range(len(self.memory_r) - 1, -1, -1):
#                 value_ = value_ * self.gamma + self.memory_r[t]
#                 discounted_r.insert(0, value_)

#             # 开始学习
#             s = self.memory_s.type(torch.FloatTensor).to(self.device)
#             a = self.memory_a.unsqueeze(-1).to(self.device)
#             r = torch.FloatTensor(discounted_r).unsqueeze(-1).to(self.device)

#             # 更新 critic
#             v = self.critic(s)
#             advantage = r - v
#             critic_loss = torch.mean(torch.pow(advantage, 2))
#             self.opt_critic.zero_grad()
#             critic_loss.backward()
#             # for param in self.critic.parameters():  # clip防止梯度爆炸
#             #     param.grad.data.clamp_(-1, 1)
#             self.opt_critic.step()

#             # 更新 actor
#             prob = self.actor(s)
#             # 选择与动作概率对应的动作
#             log_prob = torch.log(prob + 0.0001)
#             log_prob = torch.gather(log_prob, 2, a).squeeze(-1)
#             log_prob = (torch.sum(log_prob, 1).unsqueeze(-1)) / 4

#             # 计算 actor 的损失
#             actor_loss = -torch.sum(log_prob * advantage.detach())
#             self.opt_actor.zero_grad()
#             actor_loss.backward()
#             # for param in self.actor.parameters():  # clip防止梯度爆炸
#             #     param.grad.data.clamp_(-1, 1)
#             self.opt_actor.step()

#             # 重置记忆
#             self.memory_s = torch.zeros((1, self.pa.network_input_height,
#                                          self.pa.network_input_length, self.pa.network_input_width), dtype=float)
#             self.memory_a = torch.zeros((1, self.pa.num_nw))
#             self.memory_r = []
#             self.counter = 0

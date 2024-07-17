# import numpy as np

# # 环境类负责生成工作序列并管理资源和任务的状态。
# class Env: #Env 类：管理整个环境，包括生成任务序列、分配任务、计算奖励等。Env 类：生成工作序列并初始化和管理 Machine 和 JobBuffer 对象。
#     def __init__(self, pa, test_seed=101):

#         self.pa = pa
#         self.render = pa.render #是否图形化
#         self.curr_time = 0
#         self.seq_idx = 0  # 队伍中job的序号
#         self.test_seed = test_seed

#         # 产生训练还是测试数据
#         if self.pa.train:
#             np.random.seed(self.pa.seed) # 使用训练的随机种子
#         else:
#             np.random.seed(self.test_seed) # 使用训练的随机种子
#             print(self.test_seed)

#          # 持续时间和资源需求序列
#         self.nw_len_seqs, self.nw_size_seqs = self.generate_sequence_work(self.pa.simu_len * self.pa.num_nw, self.pa.num_res, self.pa.max_job_len, self.pa.max_job_size)

#         # print('self.nw_len_seqs')
#         # print(self.nw_len_seqs)
#         # print('self.nw_size_seqs')
#         # print(self.nw_size_seqs[0:3])

#         self.workload = np.zeros(pa.num_res)     ## num_res资源的数目 初始化工作负载数组    #计算此时任务序列对集群的负载
#         for i in range(pa.num_res):
#             self.workload[i] = np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs[:]) / sum(pa.cluster_res) / float(pa.simu_len)
#             # self.workload[i] = np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs[:]) /pa.cluster_res / float(pa.simu_len)
#             print("Load on # " + str(i) + " resource dimension is " + str(self.workload[i]))


#         # 初始化环境
#         self.machine = Machine(pa)
#         self.job_buffer = JobBuffer(pa)

#     def generate_sequence_work(self, simu_len, num_res, max_job_len,
#                                max_job_size):  # 定义生成任务序列的方法，参数包括模拟长度、资源数量、最大任务长度和最大任务大小

#         nw_len_seqs = np.random.randint(1, max_job_len + 1, size=(simu_len))  # 新工作对每个维度的占用时间相同 # 生成任务持续时间序列
#         # 使用 np.random.randint 生成一个长度为 simu_len 的数组，其中每个元素是从 1 到 max_job_len 之间的随机整数，表示任务的持续时间

#         nw_size_seqs = (np.random.random((simu_len, num_res)) * max_job_size).round(2)  # 生成任务资源需求序列
#         # 使用 np.random.random 生成一个大小为 (simu_len, num_res) 的二维数组，其中每个元素是 0 到 1 之间的随机小数，然后乘以 max_job_size 并四舍五入到小数点后两位，表示任务的资源需求

#         # print('nw_len_seqs')
#         # print(nw_len_seqs)
#         # print('nw_size_seqs')
#         # print(nw_size_seqs)

#         return nw_len_seqs, nw_size_seqs  # 返回生成的任务持续时间序列和任务资源需求序列

#     # 取得一个新的任务
#     def get_new_job_from_seq(self, seq_idx):   #取得一个job
#         new_job = Job(res_vec=self.nw_size_seqs[seq_idx, :],
#                       job_len=self.nw_len_seqs[seq_idx],)
#         # print(new_job)
#         return new_job

#     # 观察当前环境状态
#     def observe(self):
#         # 生成力量矩阵
#         image_repr = np.zeros(
#             (int(self.pa.network_input_height), int(self.pa.network_input_length), int(self.pa.network_input_width)),
#             dtype=float)
#         # 初始化 image_repr 矩阵，维度为 (network_input_height, network_input_length, network_input_width)，初始值为 0，数据类型为 float
#         #(2,4,5)
#         ir_pt = 0  # 初始化图像表示的指针

#         # print('self.machine.canvas')
#         # print(self.machine.canvas)

#         for i in range(self.pa.num_res):  # 可以改写成矩阵num_res=1
#             image_repr[ir_pt, :, :] = self.machine.canvas[i, :, :]
#             #canvas = [ [[14. 17. 13. 16.]      [18. 12. 15. 16.]]  ]
#             # 将机器画布的第 i 个资源矩阵复制到 image_repr 的第 ir_pt 层
#             # print(self.machine.canvas[i, :, :])
#             ir_pt += 1  # 图像表示指针加 1

#             for j in range(self.pa.num_nw):  # 遍历每一个新任务
#                 if self.job_buffer.job_buffer[j] is not None:  # 如果作业缓冲区的第 j 个作业不为空
#                     image_repr[ir_pt, :, :] = self.job_buffer.job_buffer[j].res_vec[i]
#                     # 将作业缓冲区的第 j 个作业的第 i 个资源向量复制到 image_repr 的第 ir_pt 层
#                     ir_pt += 1  # 图像表示指针加 1
#         ## image_repr 第一层（机器资源状态）：第二到第五层是四个任务分别在每个机器上的表示
#         # print('image_repr')
#         # print(image_repr)
#         # print("\n")
#         #
#         # Job(res_vec=[2.12], job_len=5)
#         # Job(res_vec=[1.81], job_len=5)
#         # Job(res_vec=[3.99], job_len=4)
#         # Job(res_vec=[4.73], job_len=1)
#         # image_repr 第一层（机器资源状态）：第二到第五层是四个任务分别在每个机器上的表示
#         # [[[14.   17.   13.   16.]
#         #   [18.   12.   15.   16.]]
#         #
#         # [[2.12  2.12  2.12  2.12]
#         # [2.12   2.12   2.12  2.12]]
#         #
#         # [[1.81  1.81  1.81  1.81]
#         #  [1.81  1.81  1.81  1.81]]
#         #
#         # [[3.99  3.99  3.99  3.99]
#         #  [3.99  3.99  3.99  3.99]]
#         #
#         # [[4.73  4.73  4.73  4.73]
#         #  [4.73  4.73  4.73  4.73]]]
#         # 返回归一化后的图像表示
#         return image_repr / self.pa.max_cluster
#         # 返回 image_repr 矩阵的归一化结果，除以最大集群值 max_cluster

#     # 计算奖励
#     def get_reward(self):
#         reward=0
#         # 声明全局变量 res_average =现在8个主机剩余带宽的平均 和 res_computer用于存储负载不平衡的方差
#         global res_average ,res_computer
#         #  canvas = [[[14. 17. 13. 16.]      [18. 12. 15. 16.]]]
#         res_average=np.mean(self.machine.canvas,(1,2),dtype=float).reshape((1,1,1))
#         # print("res_average",res_average)
#  #       res_computer=self.machine.canvas#/self.pa.cluster_res


#         var=np.sum(np.square(self.machine.canvas-res_average))/self.pa.num_clusters  #采用方差的方法
#         # 将方差赋值给 res_computer
#         res_computer = var#var方差越大越不好，所以是负权重  +1是每次完成任务
#         reward=self.pa.weight_reward*var*0.08+10
#         # print("reward",reward)

#         # res_diff=np.abs((self.machine.canvas-res_average)/self.pa.cluster_res)  #之前reward的设计方法
#         # if ((res_diff>=0.2).sum())==0:
#         #     reward+=5
#         # load_0_2=float(((res_diff<0.2).sum())*self.pa.no_balance_penalty02)
#         # load_2_4=float((((res_diff>=0.2)&(res_diff<0.4)).sum())*self.pa.no_balance_penalty24)
#         # load_4_6=float((((res_diff>=0.4)&(res_diff<0.6)).sum())*self.pa.no_balance_penalty46)
#         # load_6_10=float(((res_diff>=0.6).sum())*self.pa.no_balance_penalty610)
#         # reward+=load_2_4+load_4_6+load_6_10+load_0_2

#         return reward,res_average,res_computer

#     def step(self, action): # 执行动作 # 执行动作，返回新的状态、奖励、是否结束、调试信息

#         done = True
#         global res_average, res_computer # 使用全局变量 res_average 和 res_computer
#         # 调用 machine 的 allocate_job 方法尝试分配任务
#         # allocate_job方法用于将任务分配到资源上。如果资源充足allocated=1，则更新资源状态。
#         allocated = self.machine.allocate_job(self.job_buffer.job_buffer, action)

#     # print('allocated')
#     # print(allocated)

#         if allocated:  # 如果任务成功分配
#             # reward 是完成一次4个小任务的奖励 res_average =现在8个主机剩余带宽的平均 和 res_computer用于存储负载不平衡的方差
#             reward, res_average, res_computer = self.get_reward()  # 获取奖励和更新全局变量
#             done = False  # 设置 done 为 False，表示任务未完成
#             self.curr_time += 1  # 当前时间步增加 1
#             self.machine.time_proceed()  # 机器时间前进

#             # 添加新任务
#             self.seq_idx += 1
#             if self.seq_idx == 200:   #self.pa.simu_len-5:
#                 done = True
#                 print("序列分配完毕！")
#                 reward+= self.pa.finish_penalty

#             if not done:  # 如果任务尚未完成
#                 # 遍历num_nw=4 任务编号共200个self.seq_idx += 1  每四个任务又一组
#                 for i in range(self.pa.num_nw):
#                     # 计算真实序列索引
#                     seq_true_idx = (self.seq_idx) * self.pa.num_nw + i
#                     # print("seq_true_idx",seq_true_idx)
#                     # 从序列中获取新的作业
#                     new_job = self.get_new_job_from_seq(seq_true_idx)
#                     # print("new_job", new_job)
#                     # 将新作业放入作业缓冲区的对应位置
#                     self.job_buffer.job_buffer[i] = new_job
#                 # print("self.job_buffer.job_buffer", self.job_buffer.job_buffer)



#         else:
#             print("超出负载！只分配到第{}个序列任务（每个任务有4个流量）".format(self.seq_idx+1))
#             reward = self.pa.overload_penalty
#             done= True


#         ob = self.observe()#ob返回 image_repr （5，2，4）矩阵的归一化结果
#         return ob,reward,done,res_average,res_computer

#     # 重置环境
#     def reset(self):  # 定义重置方法
#         self.seq_idx = 0  # 将序列索引重置为0
#         self.curr_time = 0  # 将当前时间重置为0
#         #nw_len_seqs是个800长度的[4,5,2,2,1,5,....]   nw_size_seqs是800长度的带宽需求[[2.12] [1.81] [3.99] [4.73] [1.01]
#         self.nw_len_seqs, self.nw_size_seqs = self.generate_sequence_work(self.pa.simu_len * self.pa.num_nw,#200*4
#             self.pa.num_res, self.pa.max_job_len,#num_res=1表示有不同类型的带宽资源，这个参数就表示这些不同类型资源的数量。max_job_len=5单个任务可能持续的最长时间。
#             self.pa.max_job_size)  # max_job_size
#         # print("nw_len_seqs:--------",len(self.nw_len_seqs),"nw_size_seqs:-------- ",len(self.nw_size_seqs))
#         # 初始化系统
#         self.machine = Machine(self.pa)  # 创建Machine类的实例，并传入参数
#         self.job_slot = JobBuffer(self.pa)  # 创建JobBuffer类的实例，并传入参数 # 初始化任务缓冲区4个[None,None,None,None]

#         for i in range(self.pa.num_nw):  # 遍历每一个任务 self.num_nw = 4 # 每次来多少个新工作
#             seq_true_idx = (self.seq_idx) * self.pa.num_nw + i  # 计算当前作业的真实索引
#             new_job = self.get_new_job_from_seq(seq_true_idx)  # 从序列中获取新的作业
#             self.job_buffer.job_buffer[i] = new_job  # 将新作业放入作业缓冲区的相应位置  4个小任务res_vec=[0.01], job_len=1 res_vec=[3.06], job_len=3

#         for job in self.job_buffer.job_buffer:
#             print(job)
#         #ob反应 image_repr 矩阵的归一化结果image_repr 第一层（机器资源状态）：第二到第五层是四个任务分别在每个机器上的表示
#         ob = self.observe()  # 获取当前的观测状态
#         return ob, self.nw_len_seqs[:100], self.nw_size_seqs[0:100, :]  # 返回观测状态，作业长度序列的前100项和作业大小序列的前100行

# class Job:#表示一个任务，包含任务的资源需求两位浮点数和持续时间1-5。
#     def __init__(self, res_vec, job_len):
#         self.res_vec = res_vec
#         self.len = job_len

#     def __str__(self):
#         return f"Job(res_vec={self.res_vec}, job_len={self.len})"
# class JobBuffer:#管理任务缓冲区，存储当前等待分配的任务。
#     def __init__(self, pa):
#         self.job_buffer = [None] * pa.num_nw # 初始化任务缓冲区4个[None,None,None,None]

# class Machine:#管理机器资源，执行任务分配和时间推进。
#     def __init__(self, pa):
#         self.num_res = pa.num_res  # 资源数量
#         self.time_horizon = pa.time_horizon  # 时间范围
#         self.cluster_res = pa.cluster_res  # 集群资源
#         self.num_clusters = pa.num_clusters  # 集群数量
#         self.network_input_length = pa.network_input_length  # 网络输入长度
#         self.network_input_width = pa.network_input_width  # 网络输入宽度
#         self.pa = pa
#         #(11个每个8行一列)(11, 8, 1) [[1.]  [1.]  [1.]  [1.]  [1.]  [1.]  [1.]  [1.]]
#         self.avbl_res = np.ones((self.time_horizon, self.num_clusters, self.num_res))  # 初始化可用资源矩阵
#         # print("avbl_res",self.avbl_res)
#         for i in range(self.num_clusters):
#             self.avbl_res[:, i, :] *= self.cluster_res[i]#(11, 8, 1) 8个对应8个主机带宽资源
#         # 维度为（资源数量=1，网络输入长度2，网络输入宽度4）。这意味着每种资源在网络的每个部分（节点、位置）初始状态下都是可用的。
#         self.canvas = np.ones((int(pa.num_res), int(pa.network_input_length), int(pa.network_input_width)))  # 初始化画布
#         for i in range(4):
#             self.canvas[:, 0, i] *= self.cluster_res[i]
#             # print(self.canvas[:, 0, i])
#             # [14.]
#             # [17.]
#             # [13.]
#             # [16.]
#         for i in range(4):
#             self.canvas[:, 1, i] *= self.cluster_res[i + 4]
#             # print(self.canvas[:, 1, i])
#             # [18.]
#             # [12.]
#             # [15.]
#             # [16.]
#             # canvas = [[[14. 17. 13. 16.]      [18. 12. 15. 16.]]]

#     # allocate_job 方法用于将任务分配到资源上。如果资源充足，则更新资源状态。
#     def allocate_job(self, job_buffer, action):

#         # print("job_buffer")
#         # print(job_buffer[0].res_vec)
#         # print("avbl_res")
#         # print(self.avbl_res)
#         # print('job_buffer[0].len')
#         # print(job_buffer[0].len)
#         #
#         # print('action')
#         # print(action)

#         # print('self.avbl_res[:job_buffer[i].len,action[i], :]')
#         # print(self.avbl_res[:job_buffer[0].len,action[0], :])

#         allocated = False
#         for i in range(self.pa.num_nw):
#             self.avbl_res[:job_buffer[i].len,action[i], :] = self.avbl_res[:job_buffer[i].len,action[i], :] - job_buffer[i].res_vec
#         # avbl_res(11, 8, 1) 8个对应8个主机带宽资源
#         # print("avbl_res",self.avbl_res)

#         if np.all(self.avbl_res[:,:,:] >= 0):# 检查是否所有资源都满足要求

#             allocated = True

#             # 这段代码的作用是在每个时间步更新 canvas 中的资源状态，使 canvas 反映当前时间点上各路径和资源的可用状态。
#             for i in range(self.num_res):   #可以化简为矩阵运算
#                 #avbl_res 存储了各时间点、各路径上的可用资源状态(11, 8, 1)。  self.avbl_res[0,:,i],提取了当前时间点（0 代表第一个时间点）上第 i 个资源在所有路径上的状态。
#                 # np.reshape 函数将提取的资源状态重塑为一个二维矩阵，尺寸为 [self.network_input_length, self.network_input_width]，即网络输入长度和宽度。
#                 current_res = np.reshape(self.avbl_res[0,:,i],[self.network_input_length,self.network_input_width])
#                 # 更新 canvas，使其反映最新的资源状态。
#                 self.canvas[i,:,:] = current_res#current_res尺寸为 [self.network_input_length, self.network_input_width]，即网络输入长度和宽度。
#         # canvas = [[[14. 17. 13. 16.][18. 12. 15. 16.]]]
#         return allocated

#     def time_proceed(self): # time_proceed 方法模拟时间的推进，更新资源的可用状态。

#         self.avbl_res[:-1, :,:] = self.avbl_res[1:, :,:]
#     #    self.avbl_res[-1, :,:] = self.cluster_res
#         self.avbl_res[-1, :, :] = self.avbl_res[-2, :, :]
#         # 这段代码的作用是在每个时间步更新 canvas 中的资源状态，使 canvas 反映当前时间点上各路径和资源的可用状态。
#         for i in range(self.num_res):      #可以化简为矩阵运算
#                 current_res = np.reshape(self.avbl_res[0,:,i],[self.network_input_length,self.network_input_width])
#                 self.canvas[i,:,:] = current_res























import numpy as np

# 环境类负责生成工作序列并管理资源和任务的状态。
class Env: #Env 类：管理整个环境，包括生成任务序列、分配任务、计算奖励等。Env 类：生成工作序列并初始化和管理 Machine 和 JobBuffer 对象。
    def __init__(self, pa, test_seed=101):

        self.pa = pa
        self.render = pa.render #是否图形化
        self.curr_time = 0
        self.seq_idx = 0  # 队伍中job的序号
        self.test_seed = test_seed
        self.fuzai_Num=0
        self.var_list=[]

        # 产生训练还是测试数据
        if self.pa.train:
            np.random.seed(self.pa.seed) # 使用训练的随机种子
        else:
            np.random.seed(self.test_seed) # 使用训练的随机种子
            print(self.test_seed)

         # 持续时间和资源需求序列
        self.nw_len_seqs, self.nw_size_seqs = self.generate_sequence_work(self.pa.simu_len * self.pa.num_nw, self.pa.num_res, self.pa.max_job_len, self.pa.max_job_size)

        # print('self.nw_len_seqs')
        # print(self.nw_len_seqs)
        # print('self.nw_size_seqs')
        # print(self.nw_size_seqs[0:3])

        self.workload = np.zeros(pa.num_res)     ## num_res资源的数目 初始化工作负载数组    #计算此时任务序列对集群的负载
        for i in range(pa.num_res):
            self.workload[i] = np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs[:]) / sum(pa.cluster_res) / float(pa.simu_len)
            # self.workload[i] = np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs[:]) /pa.cluster_res / float(pa.simu_len)
            print("Load on # " + str(i) + " resource dimension is " + str(self.workload[i]))


        # 初始化环境
        self.machine = Machine(pa)
        self.job_buffer = JobBuffer(pa)

    def generate_sequence_work(self, simu_len, num_res, max_job_len,
                               max_job_size):  # 定义生成任务序列的方法，参数包括模拟长度、资源数量、最大任务长度和最大任务大小

        nw_len_seqs = np.random.randint(1, max_job_len + 1, size=(simu_len))  # 新工作对每个维度的占用时间相同 # 生成任务持续时间序列
        # 使用 np.random.randint 生成一个长度为 simu_len 的数组，其中每个元素是从 1 到 max_job_len 之间的随机整数，表示任务的持续时间

        nw_size_seqs = (np.random.random((simu_len, num_res)) * max_job_size).round(2)  # 生成任务资源需求序列
        # 使用 np.random.random 生成一个大小为 (simu_len, num_res) 的二维数组，其中每个元素是 0 到 1 之间的随机小数，然后乘以 max_job_size 并四舍五入到小数点后两位，表示任务的资源需求

        # print('nw_len_seqs')
        # print(nw_len_seqs)
        # print('nw_size_seqs')
        # print(nw_size_seqs)

        return nw_len_seqs, nw_size_seqs  # 返回生成的任务持续时间序列和任务资源需求序列

    # 取得一个新的任务
    def get_new_job_from_seq(self, seq_idx):   #取得一个job
        new_job = Job(res_vec=self.nw_size_seqs[seq_idx, :],
                      job_len=self.nw_len_seqs[seq_idx],)
        # print(new_job)
        return new_job

    # 观察当前环境状态
    def observe(self):
        # 生成力量矩阵
        image_repr = np.zeros(
            (int(self.pa.network_input_height), int(self.pa.network_input_length), int(self.pa.network_input_width)),
            dtype=float)
        # 初始化 image_repr 矩阵，维度为 (network_input_height, network_input_length, network_input_width)，初始值为 0，数据类型为 float
        #(2,4,5)
        ir_pt = 0  # 初始化图像表示的指针

        # print('self.machine.canvas')
        # print(self.machine.canvas)

        for i in range(self.pa.num_res):  # 可以改写成矩阵num_res=1
            image_repr[ir_pt, :, :] = self.machine.canvas[i, :, :]
            #canvas = [ [[14. 17. 13. 16.]      [18. 12. 15. 16.]]  ]
            # 将机器画布的第 i 个资源矩阵复制到 image_repr 的第 ir_pt 层
            # print(self.machine.canvas[i, :, :])
            ir_pt += 1  # 图像表示指针加 1

            for j in range(self.pa.num_nw):  # 遍历每一个新任务
                if self.job_buffer.job_buffer[j] is not None:  # 如果作业缓冲区的第 j 个作业不为空
                    image_repr[ir_pt, :, :] = self.job_buffer.job_buffer[j].res_vec[i]
                    # 将作业缓冲区的第 j 个作业的第 i 个资源向量复制到 image_repr 的第 ir_pt 层
                    ir_pt += 1  # 图像表示指针加 1
        ## image_repr 第一层（机器资源状态）：第二到第五层是四个任务分别在每个机器上的表示
        # print('image_repr')
        # print(image_repr)
        # print("\n")
        #
        # Job(res_vec=[2.12], job_len=5)
        # Job(res_vec=[1.81], job_len=5)
        # Job(res_vec=[3.99], job_len=4)
        # Job(res_vec=[4.73], job_len=1)
        # image_repr 第一层（机器资源状态）：第二到第五层是四个任务分别在每个机器上的表示
        # [[[14.   17.   13.   16.]
        #   [18.   12.   15.   16.]]
        #
        # [[2.12  2.12  2.12  2.12]
        # [2.12   2.12   2.12  2.12]]
        #
        # [[1.81  1.81  1.81  1.81]
        #  [1.81  1.81  1.81  1.81]]
        #
        # [[3.99  3.99  3.99  3.99]
        #  [3.99  3.99  3.99  3.99]]
        #
        # [[4.73  4.73  4.73  4.73]
        #  [4.73  4.73  4.73  4.73]]]
        # 返回归一化后的图像表示
        return image_repr / self.pa.max_cluster
        # 返回 image_repr 矩阵的归一化结果，除以最大集群值 max_cluster

    def get_chengfa(self):
        reward = 0
        # 声明全局变量 res_average =现在8个主机剩余带宽的平均 和 res_computer用于存储负载不平衡的方差
        global res_average, res_computer

        #  canvas = [[[14. 17. 13. 16.]      [18. 12. 15. 16.]]]
        res_average = np.mean(self.machine.canvas, (1, 2), dtype=float).reshape((1, 1, 1))
        # ---------------------------------------------------------------
        completion_reward = -30  # 超载完成一个任务。
        var = np.sum(np.square(self.machine.canvas - res_average)) / self.pa.num_clusters  # 采用方差的方法
        min_var = 0
        max_var = 50  # 35
        var = (var - min_var) / (max_var - min_var)

        # print("--------------var---------------", var)

        load_variance_penalty = -var * 5*0.1  # 方差越大，惩罚越大  #负载均衡奖励
        # utilization = np.mean(self.machine.canvas) / self.pa.max_cluster
        # utilization_reward = utilization * 5  # 利用率越高，奖励越大 #资源利用率，并根据利用率给予奖励。
        # 资源利用率奖励
        # total_used_resources已经用了的带宽资源
        # 提取第一个时间步的 (8, 1) 切片
        first_time_step_slice = self.machine.avbl_res[0, :, :]

        # # 计算第一个时间步切片的总和
        # pingjunkeyong = np.sum(first_time_step_slice)
        # # pingjunkeyong=np.sum(self.machine.avbl_res) / 11
        # # print(pingjunkeyong,"pingjunkeyong")110
        # total_used_resources = np.sum(self.pa.cluster_res) - pingjunkeyong
        # # utilization资源占用率=目前用的/总共的带宽资源
        # utilization = total_used_resources / np.sum(self.pa.cluster_res)
        # # print("utilization资源占用率：", utilization)
        # utilization_reward = round(utilization, 2) * 5
        # ---------------------------------------------------------------

        # 将方差赋值给 res_computer
        res_computer = var  # var方差越大越不好，所以是负权重  +1是每次完成任务

        # reward=self.pa.weight_reward*var*0.08+10

        # 总奖励+ utilization_reward
        reward = completion_reward + load_variance_penalty 
        # print("reward",reward)

        # res_diff=np.abs((self.machine.canvas-res_average)/self.pa.cluster_res)  #之前reward的设计方法
        # if ((res_diff>=0.2).sum())==0:
        #     reward+=5
        # load_0_2=float(((res_diff<0.2).sum())*self.pa.no_balance_penalty02)
        # load_2_4=float((((res_diff>=0.2)&(res_diff<0.4)).sum())*self.pa.no_balance_penalty24)
        # load_4_6=float((((res_diff>=0.4)&(res_diff<0.6)).sum())*self.pa.no_balance_penalty46)
        # load_6_10=float(((res_diff>=0.6).sum())*self.pa.no_balance_penalty610)
        # reward+=load_2_4+load_4_6+load_6_10+load_0_2

        return reward, res_average, res_computer
    # 计算奖励
    def get_reward(self):
        reward=0
        # 声明全局变量 res_average =现在8个主机剩余带宽的平均 和 res_computer用于存储负载不平衡的方差
        global res_average ,res_computer

        #  canvas = [[[14. 17. 13. 16.]      [18. 12. 15. 16.]]]
        res_average=np.mean(self.machine.canvas,(1,2),dtype=float).reshape((1,1,1))
        # print("----------------res_average-----------",res_average)
 #       res_computer=self.machine.canvas#/self.pa.cluster_res

        #---------------------------------------------------------------
        completion_reward = 1 # 每完成一个任务，可以给予一个固定的正奖励。
        var=np.sum(np.square(self.machine.canvas-res_average))/self.pa.num_clusters  #采用方差的方法
        
        min_var = 0#min(self.var_list)
        max_var = 50#max(self.var_list)
        # print("max_var",min_var,"max_var",max_var)
        var = (var - min_var) /max_var-min_var

        self.var_list.append(var)
        # print("----------------var_list",self.var_list)
        # var = (var - 5) 

        # print("--------------var---------------",var)

        load_variance_penalty = -var * 6*0.1 # 方差越大，惩罚越大  #负载均衡奖励
        # utilization = np.mean(self.machine.canvas) / self.pa.max_cluster
        # utilization_reward = utilization * 5  # 利用率越高，奖励越大 #资源利用率，并根据利用率给予奖励。
        # 资源利用率奖励
        #total_used_resources已经用了的带宽资源
        # 提取第一个时间步的 (8, 1) 切片
        first_time_step_slice = self.machine.avbl_res[0, :, :]

        # 计算第一个时间步切片的总和
        pingjunkeyong = np.sum(first_time_step_slice)
        # pingjunkeyong=np.sum(self.machine.avbl_res) / 11
        # print(pingjunkeyong,"pingjunkeyong")110
        total_used_resources = np.sum(self.pa.cluster_res)  - pingjunkeyong
        #utilization资源占用率=目前用的/总共的带宽资源
        utilization=total_used_resources/np.sum(self.pa.cluster_res)
        # print("-------------------------utilization资源占用率：-----------------",utilization)
        # utilization_reward= round(utilization,2)*5 
        # ---------------------------------------------------------------

        # 将方差赋值给 res_computer
        res_computer = var#var方差越大越不好，所以是负权重  +1是每次完成任务


        # reward=self.pa.weight_reward*var*0.08+10

        # 总奖励
        # reward = completion_reward + load_variance_penalty + utilization_reward
        reward = completion_reward + load_variance_penalty 
        # print("reward",reward)

        # res_diff=np.abs((self.machine.canvas-res_average)/self.pa.cluster_res)  #之前reward的设计方法
        # if ((res_diff>=0.2).sum())==0:
        #     reward+=5
        # load_0_2=float(((res_diff<0.2).sum())*self.pa.no_balance_penalty02)
        # load_2_4=float((((res_diff>=0.2)&(res_diff<0.4)).sum())*self.pa.no_balance_penalty24)
        # load_4_6=float((((res_diff>=0.4)&(res_diff<0.6)).sum())*self.pa.no_balance_penalty46)
        # load_6_10=float(((res_diff>=0.6).sum())*self.pa.no_balance_penalty610)
        # reward+=load_2_4+load_4_6+load_6_10+load_0_2

        return reward,res_average,res_computer



    def step(self, action): # 执行动作 # 执行动作，返回新的状态、奖励、是否结束、调试信息

        done = True
        global res_average, res_computer # 使用全局变量 res_average 和 res_computer
        # 调用 machine 的 allocate_job 方法尝试分配任务
        # allocate_job方法用于将任务分配到资源上。如果资源充足allocated=1，则更新资源状态。
        allocated = self.machine.allocate_job(self.job_buffer.job_buffer, action)


        if allocated:  # 如果任务成功分配
            # reward 是完成一次4个小任务的奖励 res_average =现在8个主机剩余带宽的平均 和 res_computer用于存储负载不平衡的方差
            reward, res_average, res_computer = self.get_reward()  # 获取奖励和更新全局变量
            done = False  # 设置 done 为 False，表示任务未完成
            self.curr_time += 1  # 当前时间步增加 1
            self.machine.time_proceed()  # 机器时间前进

            # 添加新任务
            self.seq_idx += 1
            if self.seq_idx == 200:   #self.pa.simu_len-5:
                done = True
                print("序列分配完毕！")
                reward+= self.pa.finish_penalty

            if not done:  # 如果任务尚未完成
                # 遍历num_nw=4 任务编号共200个self.seq_idx += 1  每四个任务又一组
                for i in range(self.pa.num_nw):
                    # 计算真实序列索引
                    seq_true_idx = (self.seq_idx) * self.pa.num_nw + i
                    # print("seq_true_idx",seq_true_idx)
                    # 从序列中获取新的作业
                    new_job = self.get_new_job_from_seq(seq_true_idx)
                    # print("new_job", new_job)
                    # 将新作业放入作业缓冲区的对应位置
                    self.job_buffer.job_buffer[i] = new_job
                # print("self.job_buffer.job_buffer", self.job_buffer.job_buffer)



        else:
            self.fuzai_Num+=1
            print("超出负载！只分配到第{}个序列任务（每个任务有4个流量）".format(self.seq_idx+1))
            reward, res_average, res_computer = self.get_chengfa()  # 获取奖励和更新全局变量


            # done = False  # 设置 done 为 False，表示任务未完成
            # self.curr_time += 1  # 当前时间步增加 1
            # self.machine.time_proceed()  # 机器时间前进

            # # 添加新任务
            # self.seq_idx += 1
            # if self.seq_idx == 200:  # self.pa.simu_len-5:
            #     done = True
            #     print("序列分配完毕！超出负载次数",self.fuzai_Num)
            #     reward += self.pa.finish_penalty#-30

            # if not done:  # 如果任务尚未完成
            #     # 遍历num_nw=4 任务编号共200个self.seq_idx += 1  每四个任务又一组
            #     for i in range(self.pa.num_nw):
            #         # 计算真实序列索引
            #         seq_true_idx = (self.seq_idx) * self.pa.num_nw + i
            #         # print("seq_true_idx",seq_true_idx)
            #         # 从序列中获取新的作业
            #         new_job = self.get_new_job_from_seq(seq_true_idx)
            #         # print("new_job", new_job)
            #         # 将新作业放入作业缓冲区的对应位置
            #         self.job_buffer.job_buffer[i] = new_job
            #     # print("self.job_buffer.job_buffer", self.job_buffer.job_buffer)


            # done= True


        ob = self.observe()#ob返回 image_repr （5，2，4）矩阵的归一化结果
        return ob,reward,done,res_average,res_computer

    # 重置环境
    def reset(self):  # 定义重置方法
        self.seq_idx = 0  # 将序列索引重置为0
        self.curr_time = 0  # 将当前时间重置为0
        #nw_len_seqs是个800长度的[4,5,2,2,1,5,....]   nw_size_seqs是800长度的带宽需求[[2.12] [1.81] [3.99] [4.73] [1.01]
        self.nw_len_seqs, self.nw_size_seqs = self.generate_sequence_work(self.pa.simu_len * self.pa.num_nw,#200*4
            self.pa.num_res, self.pa.max_job_len,#num_res=1表示有不同类型的带宽资源，这个参数就表示这些不同类型资源的数量。max_job_len=5单个任务可能持续的最长时间。
            self.pa.max_job_size)  # max_job_size
        # print("nw_len_seqs:--------",len(self.nw_len_seqs),"nw_size_seqs:-------- ",len(self.nw_size_seqs))
        # 初始化系统
        self.machine = Machine(self.pa)  # 创建Machine类的实例，并传入参数
        self.job_slot = JobBuffer(self.pa)  # 创建JobBuffer类的实例，并传入参数 # 初始化任务缓冲区4个[None,None,None,None]

        for i in range(self.pa.num_nw):  # 遍历每一个任务 self.num_nw = 4 # 每次来多少个新工作
            seq_true_idx = (self.seq_idx) * self.pa.num_nw + i  # 计算当前作业的真实索引
            new_job = self.get_new_job_from_seq(seq_true_idx)  # 从序列中获取新的作业
            self.job_buffer.job_buffer[i] = new_job  # 将新作业放入作业缓冲区的相应位置  4个小任务res_vec=[0.01], job_len=1 res_vec=[3.06], job_len=3

        # for job in self.job_buffer.job_buffer:
        #     print(job)
        #ob反应 image_repr 矩阵的归一化结果image_repr 第一层（机器资源状态）：第二到第五层是四个任务分别在每个机器上的表示
        ob = self.observe()  # 获取当前的观测状态
        return ob, self.nw_len_seqs[:200], self.nw_size_seqs[0:200, :]  # 返回观测状态，作业长度序列的前200项和作业大小序列的前100行

class Job:#表示一个任务，包含任务的资源需求两位浮点数和持续时间1-5。
    def __init__(self, res_vec, job_len):
        self.res_vec = res_vec
        self.len = job_len

    def __str__(self):
        return f"Job(res_vec={self.res_vec}, job_len={self.len})"
class JobBuffer:#管理任务缓冲区，存储当前等待分配的任务。
    def __init__(self, pa):
        self.job_buffer = [None] * pa.num_nw # 初始化任务缓冲区4个[None,None,None,None]

class Machine:#管理机器资源，执行任务分配和时间推进。
    def __init__(self, pa):
        self.num_res = pa.num_res  # 资源数量
        self.time_horizon = pa.time_horizon  # 时间范围
        self.cluster_res = pa.cluster_res  # 集群资源
        self.num_clusters = pa.num_clusters  # 集群数量
        self.network_input_length = pa.network_input_length  # 网络输入长度
        self.network_input_width = pa.network_input_width  # 网络输入宽度
        self.pa = pa
        #(11个每个8行一列)(11, 8, 1) [[1.]  [1.]  [1.]  [1.]  [1.]  [1.]  [1.]  [1.]]
        self.avbl_res = np.ones((self.time_horizon, self.num_clusters, self.num_res))  # 初始化可用资源矩阵
        for i in range(self.num_clusters):
            self.avbl_res[:, i, :] *= self.cluster_res[i]#(11, 8, 1) 8个对应8个主机带宽资源

        # 维度为（资源数量=1，网络输入长度2，网络输入宽度4）。这意味着每种资源在网络的每个部分（节点、位置）初始状态下都是可用的。
        self.canvas = np.ones((int(pa.num_res), int(pa.network_input_length), int(pa.network_input_width)))  # 初始化画布
        for i in range(4):
            self.canvas[:, 0, i] *= self.cluster_res[i]
            # print(self.canvas[:, 0, i])
            # [14.]
            # [17.]
            # [13.]
            # [16.]
        for i in range(4):
            self.canvas[:, 1, i] *= self.cluster_res[i + 4]
            # print(self.canvas[:, 1, i])
            # [18.]
            # [12.]
            # [15.]
            # [16.]
            # canvas = [[[14. 17. 13. 16.]      [18. 12. 15. 16.]]]

    # allocate_job 方法用于将任务分配到资源上。如果资源充足，则更新资源状态。
    def allocate_job(self, job_buffer, action):

        allocated = False
        for i in range(self.pa.num_nw):
            self.avbl_res[:job_buffer[i].len,action[i], :] = self.avbl_res[:job_buffer[i].len,action[i], :] - job_buffer[i].res_vec
        # avbl_res(11, 8, 1) 8个对应8个主机带宽资源
        # print("avbl_res每个时刻，每个主机的资源剩余：",self.avbl_res)

        if np.all(self.avbl_res[:,:,:] >= 0):# 检查是否所有资源都满足要求

            allocated = True

        # 这段代码的作用是在每个时间步更新 canvas 中的资源状态，使 canvas 反映当前时间点上各路径和资源的可用状态。
        for i in range(self.num_res):   #可以化简为矩阵运算
            #avbl_res 存储了各时间点、各路径上的可用资源状态(11, 8, 1)。  self.avbl_res[0,:,i],提取了当前时间点（0 代表第一个时间点）上第 i 个资源在所有路径上的状态。
            # np.reshape 函数将提取的资源状态重塑为一个二维矩阵，尺寸为 [self.network_input_length, self.network_input_width]，即网络输入长度和宽度。
            current_res = np.reshape(self.avbl_res[0,:,i],[self.network_input_length,self.network_input_width])
            # 更新 canvas，使其反映最新的资源状态。
            self.canvas[i,:,:] = current_res#current_res尺寸为 [self.network_input_length, self.network_input_width]，即网络输入长度和宽度。
        # canvas = [[[14. 17. 13. 16.][18. 12. 15. 16.]]]
        return allocated

    def time_proceed(self): # time_proceed 方法模拟时间的推进，更新资源的可用状态。

        self.avbl_res[:-1, :,:] = self.avbl_res[1:, :,:]
    #    self.avbl_res[-1, :,:] = self.cluster_res
        self.avbl_res[-1, :, :] = self.avbl_res[-2, :, :]
        # 这段代码的作用是在每个时间步更新 canvas 中的资源状态，使 canvas 反映当前时间点上各路径和资源的可用状态。
        for i in range(self.num_res):      #可以化简为矩阵运算
                current_res = np.reshape(self.avbl_res[0,:,i],[self.network_input_length,self.network_input_width])
                self.canvas[i,:,:] = current_res



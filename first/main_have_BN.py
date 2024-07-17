
import math
import random
from env import Env
import numpy as np
import pandas as pd
import torch
from Agent import A2C
import matplotlib as mpl
mpl.use('Agg')
import matplotlib
import xlwt
import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#这个代码实现了一个强化学习的集群资源管理系统，通过不断的训练和调优，来优化资源分配的策略。
class Parameters:

    def __init__(self):
        self.num_episode = 5000  # 训练轮数
        self.simu_len = 200         # 仿真序列的长度
        self.num_clusters = 8     #集群中机器的数目
        self.num_nw = 4               # 每次来多少个新工作
        self.num_res =1               # 资源的数目？ 有不同类型的带宽资源  使用不同端口用到  没用
        self.time_horizon = 11        # 集群系统所记载的时间
        self.max_job_len = 2          #  工作的最大工作时间
        self.max_job_size = 5    # 工作的最大资源需求表
        self.new_job_rate = 1        # 工作的到达概率（符合泊松分布）
        # self.cluster_res = [30, 10, 24, 30, 10, 12, 12, 40]
        self.cluster_res = [14, 17, 13, 16, 18, 12, 15, 16]         #集群中计算机每个资源的大小  #较不均匀分布
        self.max_cluster = max(self.cluster_res) #集群中资源的最大值。这个值是从 self.cluster_res 列表中计算得出的，用于标准化或其他计算。
        self.seed = 5                #用来产生不同的工作 随机种子
        self.train = True
        self.random_choice = False
        self.turn_choice = False
        self.weighted_turn = False
        self.minn_first = False#优先将任务分配给剩余资源最多的集群，以防止某些集群过载或资源利用不均衡。
        self.load_ = False  #是否加载训练好的模型
        self.save_ = True #是否保存训练好的模型
        self.close_learn = True

        self.length_ = int(math.sqrt(self.num_clusters))  # 计算集群的边长，通常用于二维网格结构的集群
        self.network_input_length = 2  # 神经网络输入的长度
        self.network_input_width = 4  # 神经网络输入的宽度
        self.network_input_height = int((self.num_res) * (1 + self.num_nw))  # 神经网络输入的高度，计算公式为 (资源数量 * (1 + 新任务数量))=5
        self.network_output_dim = int(self.num_nw)  # 神经网络输出的维度，即新任务的数量

        self.weight_reward = -0.1  # 权重奖励，负值表示惩罚
        self.no_balance_penalty02 = 0  # 不均衡的惩罚，具体在 0-2 范围内
        self.no_balance_penalty24 = -0.1  # 不均衡的惩罚，具体在 2-4 范围内
        self.no_balance_penalty46 = -0.3  # 不均衡的惩罚，具体在 4-6 范围内
        self.no_balance_penalty610 = -0.5  # 不均衡的惩罚，具体在 6-10 范围内
        self.no_balance_penalty = -0.01  # 一般不均衡的惩罚
        self.overload_penalty = -30  # 过载惩罚，当集群过载时给予较大的惩罚


        self.finish_penalty = 100  # 完成整个200任务的惩罚，设置为 0 表示没有惩罚

        # 调整探索率（用于 epsilon-greedy 策略中的探索率）
        self.min_explore = 0.01  # 最小探索率，表示算法在训练过程中最少会以 1% 的概率随机选择动作
        self.decay_min_explore = 0.995  # 探索率的衰减因子，每次更新后探索率乘以这个因子，减慢衰减速度
        self.various_epoch = 200  # 变化的周期，控制探索率变化的周期

        self.render = False  # 是否图形化显示训练过程，设置为 False 表示不显示

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备设置，优先使用 GPU

        # 学习率设置
        self.lr_actor = 1e-4  # Actor 网络的学习率，控制策略网络的更新速度
        self.lr_critic = 1e-5  # Critic 网络的学习率，控制价值网络的更新速度

        self.gamma = 0.95  # 折扣因子，表示未来奖励的折扣值，值越大表示未来奖励的影响越大
        self.max_ep_step = 2560  # 每个训练回合的最大步数，用于控制训练过程中每个回合的长度


"""----------------------------------------main函数--------------------------------"""

pa =Parameters()#初始化环境参数 pa：包含了所有的超参数和配置，如训练标志train、随机种子seed、模拟长度simu_len、资源数量num_res等。
env = Env(pa)#env包含当前环境状态、生成的工作序列、工作负载计算等。
agent = A2C(pa)#agent智能体对象，负责与环境交互，并利用神经网络进行决策和学习。
agent.test()  #没有BN层的话 会产生梯度爆炸或者梯度消失

# 如果需要加载训练好的模型
if pa.load_ == True:
    agent.load()

train_proceed =[] # 记录训练过程中的奖励
#没用
train_proceed_res_average0=[]# 未使用的变量
train_proceed_res_average1 = []  # 未使用的变量
train_proceed_res_average2 = []  # 未使用的变量

var_proceed = []  # 记录训练过程中的方差
aver_proceed = []  # 记录平均奖励
total_steps = 0  # 总步数
tot_re = 0  # 总奖励
values = [0, 1, 2, 3, 4, 5, 6, 7]  # 动作值列表
tot_crash = 0  # 总崩溃次数
res_average = 0  # 平均资源利用率

print('CUDA版本:',torch.version.cuda)
print('Pytorch版本:',torch.__version__)
print('显卡是否可用:','可用' if(torch.cuda.is_available()) else '不可用')
print('显卡数量:',torch.cuda.device_count())

if pa.weighted_turn:  # 如果使用加权轮流选择动作策略
    weiturn_action = []  # 初始化加权动作列表
    sum_wei = sum(pa.cluster_res)  # 计算所有集群资源的总和

    k = 0  # 当前集群的索引
    j = 0  # 当前集群已经加入的次数

    for i in range(sum_wei):  # 遍历总资源数次
        weiturn_action.append(k)  # 将当前集群索引 k 加入加权动作列表
        j += 1  # 增加当前集群加入的次数
        if j == pa.cluster_res[k]:  # 如果当前集群已经加入的次数等于该集群的资源数
            k += 1  # 切换到下一个集群
            j = 0  # 重置当前集群已经加入的次数

    random.shuffle(weiturn_action)  # 打乱加权动作列表

# print(weiturn_action)

for episode in range(pa.num_episode):


    print_a = []  # 动作信息列表

    print_t = []  # 时间步信息列表
    print_s = []  # 作业大小序列
    # s反应 image_repr 矩阵的归一化结果image_repr 第一层（机器资源状态）：第二到第五层是四个任务分别在每个机器上的表示
    # print_t作业长度序列的前100项和print_s作业大小序列的前100行
    s, print_t, print_s = env.reset()# 重置环境并获取初始状态和调试信息?
    # 输出初始状态、时间步信息和调试信息
    # print("Initial state (s):", s)
    # print("Time step information (print_t):", print_t)
    # print("Debug or visualization information (print_s):", print_s)

    ep_r = 0  # 回合奖励
    sum_var = 0  # 方差累加
    turn_to = 0  # 轮流选择动作的索引
    weiturn_to = 0  # 加权轮流选择动作的索引
    # var_list = []
    reword_meici=[]
    while True:
        if pa.render:
            env.render()#画图
        # interact environment

        # print("s")
        # print(s)
        s_res = []
        # for i in range(2):
        #     for j in range(4):
        #         s_res.append(s[0][i][j])

        # print("s_res")
        # print(s_res)

        # print(s[1][0][0])
        # print(s[2][0][0])

        a = [0, 0, 0, 0]
        if pa.random_choice == True:
            for i in range(pa.num_nw):
                a[i] = random.choice(values)
        elif pa.turn_choice == True:
            for i in range(pa.num_nw):
                a[i] = turn_to
                turn_to = (turn_to + 1) % pa.num_clusters
        elif pa.weighted_turn == True:
            for i in range(pa.num_nw):
                a[i] = weiturn_action[weiturn_to]
                weiturn_to = (weiturn_to+1) % sum_wei
        elif pa.minn_first == True:
            for i in range(pa.num_nw):
                j = np.argmax(s_res)#s_res 代表集群的剩余资源，np.argmax(s_res) 返回剩余资源最多的集群的索引。
                a[i] = j #分配任务 (a[i] = j)：将新任务 i 分配给选择的集群 j。
                s_res[j] -= s[i+1][0][0]
        else:
            a = agent.get_action(s)# 智s能体选择动作
            # print("a就是每一次具体任务的分配",a)

        ##s_返回 image_repr （5，2，4）矩阵的归一化结果
        # reward每一次任务得到的奖励,done
        # 全局变量 res_average =现在8个主机剩余带宽的平均 和 var是每一次任务分配后用于存储负载不平衡的方差
        s_, r,done,res_average, var = env.step(a)# 执行动作并获得新状态、奖励、完成标志、资源平均利用率和方差
        reword_meici.append(r)
        print("reword_meici--------",reword_meici)

        
        sum_var = sum_var + var

        # train_proceed_res_average.append([episode,env.seq_idx,np.squeeze(res_average)[0],np.squeeze(res_average)[1],np.squeeze(res_average)[2]])  ?????
        res_average=res_average/pa.cluster_res # 标准化资源利用率
        #train_proceed_res_average0.append([episode,env.seq_idx,np.squeeze(res_average)[0],res_diff.reshape(3,-1)[0,0],res_diff.reshape(3,-1)[0,1],res_diff.reshape(3,-1)[0,2],res_diff.reshape(3,-1)[0,3],res_diff.reshape(3,-1)[0,4],res_diff.reshape(3,-1)[0,5],res_diff.reshape(3,-1)[0,6],res_diff.reshape(3,-1)[0,7],res_diff.reshape(3,-1)[0,8],res_diff.reshape(3,-1)[0,9],res_diff.reshape(3,-1)[0,10],res_diff.reshape(3,-1)[0,11],res_diff.reshape(3,-1)[0,12],res_diff.reshape(3,-1)[0,13],res_diff.reshape(3,-1)[0,14],res_diff.reshape(3,-1)[0,15]])
        # train_proceed_res_average1.append([episode,env.seq_idx,np.squeeze(res_average)[1],res_diff.reshape(3,-1)[1,0],res_diff.reshape(3,-1)[1,1],res_diff.reshape(3,-1)[1,2],res_diff.reshape(3,-1)[1,3],res_diff.reshape(3,-1)[1,4],res_diff.reshape(3,-1)[1,5],res_diff.reshape(3,-1)[1,6],res_diff.reshape(3,-1)[1,7],res_diff.reshape(3,-1)[1,8],res_diff.reshape(3,-1)[1,9],res_diff.reshape(3,-1)[1,10],res_diff.reshape(3,-1)[1,11],res_diff.reshape(3,-1)[1,12],res_diff.reshape(3,-1)[1,13],res_diff.reshape(3,-1)[1,14],res_diff.reshape(3,-1)[1,15]])
        # train_proceed_res_average2.append([episode,env.seq_idx,np.squeeze(res_average)[2],res_diff.reshape(3,-1)[2,0],res_diff.reshape(3,-1)[2,1],res_diff.reshape(3,-1)[2,2],res_diff.reshape(3,-1)[2,3],res_diff.reshape(3,-1)[2,4],res_diff.reshape(3,-1)[2,5],res_diff.reshape(3,-1)[2,6],res_diff.reshape(3,-1)[2,7],res_diff.reshape(3,-1)[2,8],res_diff.reshape(3,-1)[2,9],res_diff.reshape(3,-1)[2,10],res_diff.reshape(3,-1)[2,11],res_diff.reshape(3,-1)[2,12],res_diff.reshape(3,-1)[2,13],res_diff.reshape(3,-1)[2,14],res_diff.reshape(3,-1)[2,15]])

        if pa.random_choice == False and pa.turn_choice == False and pa.close_learn == False:
            agent.learn(s, a, s_, r, done)# 学习过程
        # update record
        ep_r += r # 累加回合奖励
        s = s_# 更新状态
        total_steps += 1# 增加总步数

        if done:#超载或者是200个任务都完成了
            train_proceed.append(ep_r) # 超载或者是200个任务完成 记录一论训练的所有任务奖励
            # 记录方差（每一次训练的方差相加共200次/200）（没完成：每一次训练的方差相加/次数）
            var_proceed.append(sum_var/env.seq_idx)
            print('episode: ', episode,
                  'ep_r: ', round(ep_r, 2),
                  "序列",env.seq_idx )
            if env.seq_idx != pa.simu_len:
                tot_crash += 1
            # tot_re += ep_r
            #
            # if episode % 20 == 0:
            #     print("\n平均数:", tot_re/20)
            #     tot_re = 0
            break
# 如果需要保存训练好的模型
if pa.train == True and pa.save_ == True:
    agent.save()
per_aver = 20  # 定义平均奖励的周期，即每20次计算一个平均奖励
list_ = []  # 初始化一个空列表，用于存储x轴的刻度值
aver_proceed = []  # 初始化一个空列表，用于存储每个周期的平均奖励
# 计算平均奖励
for i in range(pa.num_episode):  # 遍历所有的训练轮数
    if i % per_aver == 0:  # 如果当前轮数是平均周期的倍数
        aver_proceed.append(0)  # 在平均奖励列表中添加一个0，表示新的周期开始
        list_.append(0)  # 在x轴刻度列表中添加一个0

    # 累加当前周期内的奖励   每20个训练轮数打印一个点用于作图
    aver_proceed[int(i / per_aver)] = aver_proceed[int(i / per_aver)] + train_proceed[i]#训练20轮的奖励总和
    # print("aver_proceed",aver_proceed)
    if i % per_aver == per_aver - 1:  # 如果当前轮数是平均周期的末尾
        # 计算当前周期的平均奖励
        aver_proceed[int(i / per_aver)] = aver_proceed[int(i / per_aver)] / per_aver
        # print("aver_proceed---------------------",aver_proceed)
        list_[int(i / per_aver)] = i  # 更新x轴刻度列表中当前周期的末尾索引

# 绘制训练过程中的奖励曲线
plt.figure(num=1,figsize=(4,4))       #显示框设置(一个该函数，就是一个单另的显示框)
plt.xlim(0,pa.num_episode+1)
plt.ylim(-20,500)
plt.plot(train_proceed,color='blue',linewidth=0.5)      #画图设置
plt.xlabel('episode')
plt.ylabel('reward')
plt.savefig( 'fig_ex_unb_minn_first.1.png')
# plt.show() 
plt.close()    #显示图片（类似于print())）
# 绘制平均奖励曲线
plt.plot(list_, aver_proceed, color='orange',linewidth=3.0)
plt.xlim(0,pa.num_episode+1)
plt.ylim(-40,500)
plt.xlabel('episode')
# 每20轮训练的average reward
plt.ylabel('average reward')
plt.savefig( 'fig_ex_unb__minn_first.2.png')
# plt.show()     #显示图片（类似于print())）
plt.close()  
# 绘制方差曲线
plt.plot(var_proceed,color='green',linewidth=0.5)
plt.xlim(0,pa.num_episode+1)
plt.ylim(0,1)
plt.xlabel('episode')
# 每1轮包含200个任务，但不一定完成200个任务的训练方差var
plt.ylabel('var')
plt.savefig( 'fig_ex_unb__minn_first.3.png')
# plt.show()
plt.close()  

# print("Load 0.30")
# if pa.random_choice:
#     print("random")
# elif pa.turn_choice:
#     print("turn")
# else :
#     print("Rl")
#
# print("aver_reward")
# aver_reward_ = sum(train_proceed) / pa.num_episode
# print(aver_reward_.round(2))
#
# print("aver_var")
# #aver_var = sum(var_proceed) / pa.num_episode
# aver_var = 0
# for i in range(50):
#     aver_var += var_proceed[i]
# print(aver_var.round(4))
#
# print("crash_rating")
# rating_ = tot_crash / pa.num_episode
# print(rating_)
print("print_s--------------------------")
print(print_s)
print("print_t---------------------------")
print(print_t)
print("print_a---------------------------")
print(print_a)

# 创建一个新的 Excel 工作簿
workbook = xlwt.Workbook()
sheet = workbook.add_sheet('stream & last_time & action')
# 将调试信息写入 Excel
for i in range(100):
    for j in range(4):
        # sheet.write(i, j, str(print_a[i][j]))  改
        sheet.write(i, j+4, str(print_t[j+i*4]))
        sheet.write(i, j+8, str(print_s[j+i*4][0]))
# 保存 Excel 文件
workbook.save('Excel1.xls')


# #3 设置刻度及步长
# z = range(40)
# x_label = ['11:{}'.format(i) for i in x]
# plt.xticks( x[::5], x_label[::5])
# plt.yticks(z[::5])  #5是步长
#
# #4 添加网格信息
# plt.grid(True, linestyle='--', alpha=0.5) #默认是True，风格设置为虚线，alpha为透明度
#
# #5 添加标题（中文在plt中默认乱码，不乱码的方法在本文最后说明）
# plt.xlabel('Time')
# plt.ylabel('Temperature')
# plt.title('Curve of Temperature Change with Time')
#
# #6 保存图片，并展示
# plt.savefig('./plt_png/test1.2.png')
# plt.show()


#
#
# train_proceed=np.array(train_proceed)
#
# # ##用来打印累计奖励
# columns = ["epoch", "reward"]
# dt = pd.DataFrame(train_proceed, columns=columns)
# dt.to_excel("learning_reward{}_0999_BN_4long.xlsx".format(pa.num_episode), index=0)
# dt.to_csv("learning_reward{}_0999_BN_4long.csv".format(pa.num_episode), index=0)
#
#
#
# #用来打印每个资源每一轮的均值
# columns = ["epoch", "seq_index","aver","load_per_computer1","load_per_computer2","load_per_computer3","load_per_computer4","load_per_computer5","load_per_computer6","load_per_computer7","load_per_computer8","load_per_computer9","load_per_computer10","load_per_computer11","load_per_computer12","load_per_computer13","load_per_computer14","load_per_computer15","load_per_computer16"]
# for i in range(3):
#     exec("dt{} = pd.DataFrame(train_proceed_res_average{}, columns=columns)".format(i,i))
#     # exec("dt{}.to_excel('learning_reward{}_{}.xlsx', index=0)".format(i,i,pa.num_episode))
#     exec("dt{}.to_csv('learning_reward{}_{}_BN_4long.csv', index=0)".format(i,i,pa.num_episode))







# import math
# import random
# from env import Env
# import numpy as np
# import pandas as pd
# import torch
# from Agent import A2C
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib
# import xlwt
# import matplotlib.pyplot as plt
# # matplotlib.use('Qt5Agg')
# #这个代码实现了一个强化学习的集群资源管理系统，通过不断的训练和调优，来优化资源分配的策略。
# class Parameters:

#     def __init__(self):
#         self.num_episode = 20000  # 训练轮数
#         self.simu_len = 200         # 仿真序列的长度
#         self.num_clusters = 8     #集群中机器的数目
#         self.num_nw = 4               # 每次来多少个新工作
#         self.num_res = 1               # 资源的数目？ 有不同类型的带宽资源
#         self.time_horizon = 11        # 集群系统所记载的时间
#         self.max_job_len = 5          #  工作的最大工作时间
#         self.max_job_size = 5    # 工作的最大资源需求表
#         self.new_job_rate = 1        # 工作的到达概率（符合泊松分布）
#     #    self.cluster_res = [5, 1, 24, 30, 6, 12, 3, 40]
#         self.cluster_res = [14, 17, 13, 16, 18, 12, 15, 16]         #集群中计算机每个资源的大小  #较不均匀分布
#         self.max_cluster = max(self.cluster_res)#集群中资源的最大值。这个值是从 self.cluster_res 列表中计算得出的，用于标准化或其他计算。
#         self.seed = 5                #用来产生不同的工作 随机种子
#         self.train = True
#         self.random_choice = False
#         self.turn_choice = False
#         self.weighted_turn = False
#         self.minn_first = False#优先将任务分配给剩余资源最多的集群，以防止某些集群过载或资源利用不均衡。
#         self.load_ = False  #是否加载训练好的模型
#         self.save_ = False #是否保存训练好的模型
#         self.close_learn = True

#         self.length_ = int(math.sqrt(self.num_clusters))  # 计算集群的边长，通常用于二维网格结构的集群
#         self.network_input_length = 2  # 神经网络输入的长度
#         self.network_input_width = 4  # 神经网络输入的宽度
#         self.network_input_height = int((self.num_res) * (1 + self.num_nw))  # 神经网络输入的高度，计算公式为 (资源数量 * (1 + 新任务数量))=5
#         self.network_output_dim = int(self.num_nw)  # 神经网络输出的维度，即新任务的数量

#         self.weight_reward = -0.1  # 权重奖励，负值表示惩罚
#         self.no_balance_penalty02 = 0  # 不均衡的惩罚，具体在 0-2 范围内
#         self.no_balance_penalty24 = -0.1  # 不均衡的惩罚，具体在 2-4 范围内
#         self.no_balance_penalty46 = -0.3  # 不均衡的惩罚，具体在 4-6 范围内
#         self.no_balance_penalty610 = -0.5  # 不均衡的惩罚，具体在 6-10 范围内
#         self.no_balance_penalty = -0.01  # 一般不均衡的惩罚
#         self.overload_penalty = -10  # 过载惩罚，当集群过载时给予较大的惩罚
#         self.finish_penalty = 100  # 完成整个200任务的惩罚，设置为 0 表示没有惩罚

#         # 调整探索率（用于 epsilon-greedy 策略中的探索率）
#         self.min_explore = 0.01  # 最小探索率，表示算法在训练过程中最少会以 1% 的概率随机选择动作
#         self.decay_min_explore = 0.995  # 探索率的衰减因子，每次更新后探索率乘以这个因子，减慢衰减速度
#         self.various_epoch = 200  # 变化的周期，控制探索率变化的周期

#         self.render = False  # 是否图形化显示训练过程，设置为 False 表示不显示

#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备设置，优先使用 GPU

#         # 学习率设置
#         self.lr_actor = 1e-4  # Actor 网络的学习率，控制策略网络的更新速度
#         self.lr_critic = 1e-5  # Critic 网络的学习率，控制价值网络的更新速度

#         self.gamma = 0.9  # 折扣因子，表示未来奖励的折扣值，值越大表示未来奖励的影响越大
#         self.max_ep_step = 2560  # 每个训练回合的最大步数，用于控制训练过程中每个回合的长度


# """----------------------------------------main函数--------------------------------"""

# pa =Parameters()#初始化环境参数 pa：包含了所有的超参数和配置，如训练标志train、随机种子seed、模拟长度simu_len、资源数量num_res等。
# env = Env(pa)#env包含当前环境状态、生成的工作序列、工作负载计算等。
# agent = A2C(pa)#agent智能体对象，负责与环境交互，并利用神经网络进行决策和学习。
# agent.test()  #没有BN层的话 会产生梯度爆炸或者梯度消失

# # 如果需要加载训练好的模型
# if pa.load_ == True:
#     agent.load()

# train_proceed =[] # 记录训练过程中的奖励
# #没用
# train_proceed_res_average0=[]# 未使用的变量
# train_proceed_res_average1 = []  # 未使用的变量
# train_proceed_res_average2 = []  # 未使用的变量

# var_proceed = []  # 记录训练过程中的方差
# aver_proceed = []  # 记录平均奖励
# total_steps = 0  # 总步数
# tot_re = 0  # 总奖励
# values = [0, 1, 2, 3, 4, 5, 6, 7]  # 动作值列表
# tot_crash = 0  # 总崩溃次数
# res_average = 0  # 平均资源利用率

# print('CUDA版本:',torch.version.cuda)
# print('Pytorch版本:',torch.__version__)
# print('显卡是否可用:','可用' if(torch.cuda.is_available()) else '不可用')
# print('显卡数量:',torch.cuda.device_count())

# if pa.weighted_turn:  # 如果使用加权轮流选择动作策略
#     weiturn_action = []  # 初始化加权动作列表
#     sum_wei = sum(pa.cluster_res)  # 计算所有集群资源的总和

#     k = 0  # 当前集群的索引
#     j = 0  # 当前集群已经加入的次数

#     for i in range(sum_wei):  # 遍历总资源数次
#         weiturn_action.append(k)  # 将当前集群索引 k 加入加权动作列表
#         j += 1  # 增加当前集群加入的次数
#         if j == pa.cluster_res[k]:  # 如果当前集群已经加入的次数等于该集群的资源数
#             k += 1  # 切换到下一个集群
#             j = 0  # 重置当前集群已经加入的次数

#     random.shuffle(weiturn_action)  # 打乱加权动作列表

# # print(weiturn_action)

# for episode in range(pa.num_episode):


#     print_a = []  # 动作信息列表

#     print_t = []  # 时间步信息列表
#     print_s = []  # 作业大小序列
#     # s反应 image_repr 矩阵的归一化结果image_repr 第一层（机器资源状态）：第二到第五层是四个任务分别在每个机器上的表示
#     # print_t作业长度序列的前100项和print_s作业大小序列的前100行
#     s, print_t, print_s = env.reset()# 重置环境并获取初始状态和调试信息?
#     # 输出初始状态、时间步信息和调试信息
#     # print("Initial state (s):", s)
#     # print("Time step information (print_t):", print_t)
#     # print("Debug or visualization information (print_s):", print_s)

#     ep_r = 0  # 回合奖励
#     sum_var = 0  # 方差累加
#     turn_to = 0  # 轮流选择动作的索引
#     weiturn_to = 0  # 加权轮流选择动作的索引

#     while True:
#         if pa.render:
#             env.render()#画图
#         # interact environment

#         # print("s")
#         # print(s)
#         s_res = []
#         # for i in range(2):
#         #     for j in range(4):
#         #         s_res.append(s[0][i][j])

#         # print("s_res")
#         # print(s_res)

#         # print(s[1][0][0])
#         # print(s[2][0][0])

#         a = [0, 0, 0, 0]
#         if pa.random_choice == True:
#             for i in range(pa.num_nw):
#                 a[i] = random.choice(values)
#         elif pa.turn_choice == True:
#             for i in range(pa.num_nw):
#                 a[i] = turn_to
#                 turn_to = (turn_to + 1) % pa.num_clusters
#         elif pa.weighted_turn == True:
#             for i in range(pa.num_nw):
#                 a[i] = weiturn_action[weiturn_to]
#                 weiturn_to = (weiturn_to+1) % sum_wei
#         elif pa.minn_first == True:
#             for i in range(pa.num_nw):
#                 j = np.argmax(s_res)#s_res 代表集群的剩余资源，np.argmax(s_res) 返回剩余资源最多的集群的索引。
#                 a[i] = j#分配任务 (a[i] = j)：将新任务 i 分配给选择的集群 j。
#                 s_res[j] -= s[i+1][0][0]
#         else:
#             a = agent.get_action(s)# 智s能体选择动作

#         #
#         # print("a")
#         # print(a)
#         ##s_返回 image_repr （5，2，4）矩阵的归一化结果
#         # reward每一次任务得到的奖励,done
#         # 全局变量 res_average =现在8个主机剩余带宽的平均 和 var是每一次任务分配后用于存储负载不平衡的方差
#         s_, r,done,res_average, var = env.step(a)# 执行动作并获得新状态、奖励、完成标志、资源平均利用率和方差
#         sum_var = sum_var + var

#         # train_proceed_res_average.append([episode,env.seq_idx,np.squeeze(res_average)[0],np.squeeze(res_average)[1],np.squeeze(res_average)[2]])  ?????
#         res_average=res_average/pa.cluster_res # 标准化资源利用率
#         #train_proceed_res_average0.append([episode,env.seq_idx,np.squeeze(res_average)[0],res_diff.reshape(3,-1)[0,0],res_diff.reshape(3,-1)[0,1],res_diff.reshape(3,-1)[0,2],res_diff.reshape(3,-1)[0,3],res_diff.reshape(3,-1)[0,4],res_diff.reshape(3,-1)[0,5],res_diff.reshape(3,-1)[0,6],res_diff.reshape(3,-1)[0,7],res_diff.reshape(3,-1)[0,8],res_diff.reshape(3,-1)[0,9],res_diff.reshape(3,-1)[0,10],res_diff.reshape(3,-1)[0,11],res_diff.reshape(3,-1)[0,12],res_diff.reshape(3,-1)[0,13],res_diff.reshape(3,-1)[0,14],res_diff.reshape(3,-1)[0,15]])
#         # train_proceed_res_average1.append([episode,env.seq_idx,np.squeeze(res_average)[1],res_diff.reshape(3,-1)[1,0],res_diff.reshape(3,-1)[1,1],res_diff.reshape(3,-1)[1,2],res_diff.reshape(3,-1)[1,3],res_diff.reshape(3,-1)[1,4],res_diff.reshape(3,-1)[1,5],res_diff.reshape(3,-1)[1,6],res_diff.reshape(3,-1)[1,7],res_diff.reshape(3,-1)[1,8],res_diff.reshape(3,-1)[1,9],res_diff.reshape(3,-1)[1,10],res_diff.reshape(3,-1)[1,11],res_diff.reshape(3,-1)[1,12],res_diff.reshape(3,-1)[1,13],res_diff.reshape(3,-1)[1,14],res_diff.reshape(3,-1)[1,15]])
#         # train_proceed_res_average2.append([episode,env.seq_idx,np.squeeze(res_average)[2],res_diff.reshape(3,-1)[2,0],res_diff.reshape(3,-1)[2,1],res_diff.reshape(3,-1)[2,2],res_diff.reshape(3,-1)[2,3],res_diff.reshape(3,-1)[2,4],res_diff.reshape(3,-1)[2,5],res_diff.reshape(3,-1)[2,6],res_diff.reshape(3,-1)[2,7],res_diff.reshape(3,-1)[2,8],res_diff.reshape(3,-1)[2,9],res_diff.reshape(3,-1)[2,10],res_diff.reshape(3,-1)[2,11],res_diff.reshape(3,-1)[2,12],res_diff.reshape(3,-1)[2,13],res_diff.reshape(3,-1)[2,14],res_diff.reshape(3,-1)[2,15]])

#         if pa.random_choice == False and pa.turn_choice == False and pa.close_learn == False:
#             agent.learn(s, a, s_, r, done)# 学习过程
#         # update record
#         ep_r += r # 累加回合奖励
#         s = s_# 更新状态
#         total_steps += 1# 增加总步数

#         if done:#超载或者是200个任务都完成了
#             train_proceed.append(ep_r) # 超载或者是200个任务完成 记录总回合奖励
#             # 记录方差（每一次训练的方差相加共200次/200）（没完成：每一次训练的方差相加/次数）
#             var_proceed.append(sum_var/env.seq_idx)
#             print('episode: ', episode,
#                   'ep_r: ', round(ep_r, 2),
#                   "序列",env.seq_idx )
#             if env.seq_idx != pa.simu_len:
#                 tot_crash += 1
#             # tot_re += ep_r
#             #
#             # if episode % 20 == 0:
#             #     print("\n平均数:", tot_re/20)
#             #     tot_re = 0
#             break
# # 如果需要保存训练好的模型
# if pa.train == True and pa.save_ == True:
#     agent.save()
# per_aver = 20  # 定义平均奖励的周期，即每20次计算一个平均奖励
# list_ = []  # 初始化一个空列表，用于存储x轴的刻度值
# aver_proceed = []  # 初始化一个空列表，用于存储每个周期的平均奖励
# # 计算平均奖励
# for i in range(pa.num_episode):  # 遍历所有的训练轮数
#     if i % per_aver == 0:  # 如果当前轮数是平均周期的倍数
#         aver_proceed.append(0)  # 在平均奖励列表中添加一个0，表示新的周期开始
#         list_.append(0)  # 在x轴刻度列表中添加一个0

#     # 累加当前周期内的奖励
#     aver_proceed[int(i / per_aver)] = aver_proceed[int(i / per_aver)] + train_proceed[i]
#     print("aver_proceed",aver_proceed)
#     if i % per_aver == per_aver - 1:  # 如果当前轮数是平均周期的末尾
#         # 计算当前周期的平均奖励
#         aver_proceed[int(i / per_aver)] = aver_proceed[int(i / per_aver)] / per_aver
#         list_[int(i / per_aver)] = i  # 更新x轴刻度列表中当前周期的末尾索引

# # 绘制训练过程中的奖励曲线
# plt.figure(num=1,figsize=(4,4))       #显示框设置(一个该函数，就是一个单另的显示框)
# plt.xlim(0,pa.num_episode+1)
# plt.ylim(-20,1000)
# plt.plot(train_proceed,color='blue',linewidth=0.5)      #画图设置
# plt.xlabel('episode')
# plt.ylabel('reward')
# plt.savefig( 'fig_ex_unb_minn_first.1.png')
# plt.show()     #显示图片（类似于print())）
# # 绘制平均奖励曲线
# plt.plot(list_, aver_proceed, color='orange',linewidth=3.0)
# plt.xlim(0,pa.num_episode+1)
# plt.ylim(-20,1000)
# plt.xlabel('episode')
# plt.ylabel('average reward')
# plt.savefig( 'fig_ex_unb__minn_first.2.png')
# plt.show()     #显示图片（类似于print())）
# # 绘制方差曲线
# plt.plot(var_proceed,color='green',linewidth=0.5)
# plt.xlim(0,pa.num_episode+1)
# plt.ylim(0,50)
# plt.xlabel('episode')
# plt.ylabel('var')
# plt.savefig( 'fig_ex_unb__minn_first.3.png')
# plt.show()

# # print("Load 0.30")
# # if pa.random_choice:
# #     print("random")
# # elif pa.turn_choice:
# #     print("turn")
# # else :
# #     print("Rl")
# #
# # print("aver_reward")
# # aver_reward_ = sum(train_proceed) / pa.num_episode
# # print(aver_reward_.round(2))
# #
# # print("aver_var")
# # #aver_var = sum(var_proceed) / pa.num_episode
# # aver_var = 0
# # for i in range(50):
# #     aver_var += var_proceed[i]
# # print(aver_var.round(4))
# #
# # print("crash_rating")
# # rating_ = tot_crash / pa.num_episode
# # print(rating_)
# print("print_s--------------------------")
# print(print_s)
# print("print_t---------------------------")
# print(print_t)
# print("print_a---------------------------")
# print(print_a)

# # 创建一个新的 Excel 工作簿
# workbook = xlwt.Workbook()
# sheet = workbook.add_sheet('stream & last_time & action')
# # 将调试信息写入 Excel
# for i in range(200):
#     for j in range(4):
#         # sheet.write(i, j, str(print_a[i][j]))  改
#         sheet.write(i, j+4, str(print_t[j+i*4]))
#         sheet.write(i, j+8, str(print_s[j+i*4][0]))
# # 保存 Excel 文件
# workbook.save('Excel1.xls')


# # #3 设置刻度及步长
# # z = range(40)
# # x_label = ['11:{}'.format(i) for i in x]
# # plt.xticks( x[::5], x_label[::5])
# # plt.yticks(z[::5])  #5是步长
# #
# # #4 添加网格信息
# # plt.grid(True, linestyle='--', alpha=0.5) #默认是True，风格设置为虚线，alpha为透明度
# #
# # #5 添加标题（中文在plt中默认乱码，不乱码的方法在本文最后说明）
# # plt.xlabel('Time')
# # plt.ylabel('Temperature')
# # plt.title('Curve of Temperature Change with Time')
# #
# # #6 保存图片，并展示
# # plt.savefig('./plt_png/test1.2.png')
# # plt.show()


# #
# #
# # train_proceed=np.array(train_proceed)
# #
# # # ##用来打印累计奖励
# # columns = ["epoch", "reward"]
# # dt = pd.DataFrame(train_proceed, columns=columns)
# # dt.to_excel("learning_reward{}_0999_BN_4long.xlsx".format(pa.num_episode), index=0)
# # dt.to_csv("learning_reward{}_0999_BN_4long.csv".format(pa.num_episode), index=0)
# #
# #
# #
# # #用来打印每个资源每一轮的均值
# # columns = ["epoch", "seq_index","aver","load_per_computer1","load_per_computer2","load_per_computer3","load_per_computer4","load_per_computer5","load_per_computer6","load_per_computer7","load_per_computer8","load_per_computer9","load_per_computer10","load_per_computer11","load_per_computer12","load_per_computer13","load_per_computer14","load_per_computer15","load_per_computer16"]
# # for i in range(3):
# #     exec("dt{} = pd.DataFrame(train_proceed_res_average{}, columns=columns)".format(i,i))
# #     # exec("dt{}.to_excel('learning_reward{}_{}.xlsx', index=0)".format(i,i,pa.num_episode))
# #     exec("dt{}.to_csv('learning_reward{}_{}_BN_4long.csv', index=0)".format(i,i,pa.num_episode))

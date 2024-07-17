import torch
import torch.nn as nn
# 实现A2C算法，包括Actor和Critic两个网络，以及训练逻辑。
class Critic(nn.Module):
    # Critic Network
    # input: the state and corresponding action
    # output: the q value
    def __init__(self):
        super(Critic,self).__init__()
        self.ker_size = 1
        self.canvas_padding = 0
        self.conv1 = nn.Sequential(         # input shape (batchsize，15, 4, 4)
            nn.Conv2d(
                in_channels=5,              # input height
                out_channels=1000,            # n_filters
                kernel_size=self.ker_size,              # filter size
                stride=1,                   # filter movement/step
                padding=self.canvas_padding,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),     
            nn.BatchNorm2d(1000,eps=1e-5,momentum=0.1)   ,                      # output shape (batchsize，32, 2, 4)
            nn.ReLU(),                      # activation
        )
        self.conv1_1=nn.Sequential(
            nn.Conv2d(1000,500,self.ker_size,1,self.canvas_padding),
            nn.BatchNorm2d(500,eps=1e-5,momentum=0.1)   ,
            nn.ReLU()
        )
        self.conv1_2=nn.Sequential(
            nn.Conv2d(500,100,self.ker_size,1,self.canvas_padding),
            nn.BatchNorm2d(100,eps=1e-5,momentum=0.1)   ,
            nn.ReLU()
        )
        self.conv1_3=nn.Sequential(
            nn.Conv2d(100,50,self.ker_size,1,self.canvas_padding),
            nn.BatchNorm2d(50,eps=1e-5,momentum=0.1)   ,
            nn.ReLU()
        )
        self.conv1_4=nn.Sequential(
            nn.Conv2d(50,4,self.ker_size,1,self.canvas_padding),
            nn.BatchNorm2d(4,eps=1e-5,momentum=0.1)   ,
            nn.ReLU()
        )
         
        self.input_x = nn.Linear(4*2*4, 16)
        # self.bn=nn.BatchNorm1d(16,eps=1e-5,momentum=0.1)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv1_1(x)
        x=self.conv1_2(x)
        x=self.conv1_3(x)
        x=self.conv1_4(x)  
              
  
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 4 * 2 * 4)

        # print("xxxxxxxxx")
        # print(x)

        input_x=self.input_x(x) 
        # input_x=self.bn(input_x)
        input_total = torch.relu(input_x)
        q = self.output(input_total)
        return q    #batch 1


class Actor(nn.Module):
    # Action Network
    # input: the state
    # output: the action
    def __init__(self):
        super(Actor,self).__init__()
        self.ker_size = 1
        self.canvas_padding = 0
        self.conv1 = nn.Sequential(         # input shape (batchsize，15, 4, 4)
            nn.Conv2d(
                in_channels=5,              # input height
                out_channels=1000,            # n_filters
                kernel_size=self.ker_size,              # filter size
                stride=1,                   # filter movement/step
                padding=self.canvas_padding,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (batchsize，32, 4, 4)
            nn.BatchNorm2d(1000,eps=1e-5,momentum=0.1)   ,    
            nn.ReLU(),                      # activation
        )
        self.conv1_1=nn.Sequential(
            nn.Conv2d(1000,500,self.ker_size,1,self.canvas_padding),
            nn.BatchNorm2d(500,eps=1e-5,momentum=0.1)   ,    
            nn.ReLU()
        )
        self.conv1_2=nn.Sequential(
            nn.Conv2d(500,100,self.ker_size,1,self.canvas_padding),
            nn.BatchNorm2d(100,eps=1e-5,momentum=0.1)   ,    
            nn.ReLU()
        )
        self.conv1_3=nn.Sequential(
            nn.Conv2d(100,50,self.ker_size,1,self.canvas_padding),
            nn.BatchNorm2d(50,eps=1e-5,momentum=0.1)   ,    
            nn.ReLU()
        )
        self.conv1_4=nn.Sequential(
            nn.Conv2d(50,4,self.ker_size,1,self.canvas_padding),
            # nn.BatchNorm2d(4,momentum=0.1)   ,
            nn.ReLU()
        )      

        self.net_1 = nn.Softmax(dim=2)


    def forward(self, x):

        # print("x")
        # print(x)

        x=self.conv1(x)
        x=self.conv1_1(x)
        x=self.conv1_2(x)
        x=self.conv1_3(x)
        x=self.conv1_4(x)           

        a=x.size(0)
        b=x.size(1)

        # print('a')
        # print(a)
        # print('b')
        # print(b)

        # print("x1")
        # print(x)

        x = x.view(a, b,-1)

        # print("x2")
        # print(x)

        x=self.net_1(x)     # match  4  16

        # print("x3")
        # print(x)

        return x
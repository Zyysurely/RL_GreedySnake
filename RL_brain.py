import numpy as np
import tensorflow as tf
from collections import deque
import random


np.random.seed(1)
tf.set_random_seed(1)

observe=60

# Deep Q Network off-policy  input是[80, 80, 1]
class DeepQNetwork:
    def __init__(self, 
            n_actions,
            model_path,
            replay_memory=deque(),
            current_timestep=0,
            explore=200000.,
            initial_epsilon=0.9,
            final_epsilon=0.1,
            gamma=0.99,
            replay_size=50000,
            batch_size=32,
            output_graph=True    # 是否输出图
    ):
        # 参数设定
        self.n_actions = n_actions              # action的大小
        self.GAMMA = gamma                      # learning rate
        self.OBSERVE = observe                  # timesteps to observe before training，一开始先不训练观察，有足够训练数据后再开始
        self.EXPLORE = explore                  
        self.FINAL_EPSILON = final_epsilon      # final value of epsilon
        self.INITIAL_EPSILON = initial_epsilon  # starting value of epsilon
        self.REPLAY_SIZE = replay_size          # 记忆库的大小
        self.BATCH_SIZE = batch_size            # 每个训练的batch大小
        self.UPDATE_TIME = 100                  # 更新target网络的步数
        self.learning_rate = 1e-3               # nn的learning rate
        self.cost = 0                           # loss

        self.replayMemory = replay_memory

        self.timestep = 0
        self.initial_timestep=current_timestep
        self.accual_timestep=self.initial_timestep+self.timestep 
        #for loading mode
        # self.epsilon = self.INITIAL_EPSILON-(self.INITIAL_EPSILON-self.FINAL_EPSILON)/self.EXPLORE*self.accual_timestep
        # if self.epsilon<self.FINAL_EPSILON:
        #     self.epsilon=self.FINAL_EPSILON
        self.epsilon = 0.2
        
        self._build_net()
        self.session = tf.InteractiveSession()
        
        # 存储文件存储到tensorboard
        if(output_graph):
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("./logs/", self.session.graph)
        
        self.session.run(tf.global_variables_initializer())
        self.cost_his = []
    
    # 搭建整个网络和loss
    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.stateInput = tf.placeholder("float", [None, 80, 80, 4])
        self.stateInputT = tf.placeholder("float", [None, 80, 80, 4])
        self.actionInput = tf.placeholder("float", [None, self.n_actions])
        self.rewardInput = tf.placeholder("float", [None])

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope("eval_net"):
            self.QValue, \
            self.W_conv1, \
            self.b_conv1, \
            self.W_conv2, \
            self.b_conv2, \
            self.W_conv3, \
            self.b_conv3, \
            self.W_fc1, \
            self.b_fc1, \
            self.W_fc2, \
            self.b_fc2 = self._cov_cnn(self.stateInput)
        
        # ------------------ build target_net ------------------
        with tf.variable_scope("target_net"):
            self.QValueT,\
            self.W_conv1T,\
            self.b_conv1T,\
            self.W_conv2T,\
            self.b_conv2T,\
            self.W_conv3T,\
            self.b_conv3T,\
            self.W_fc1T,\
            self.b_fc1T,\
            self.W_fc2T,\
            self.b_fc2T = self._cov_cnn(self.stateInputT)
        
        # 更新target网络的参数
        with tf.variable_scope('hard_replacement'):
            self.copyTargetQNetworkOperation = \
                [self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3), self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2)]
        with tf.variable_scope('q_eval'):
            Q_eval = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices=1)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.rewardInput, Q_eval), name='TD_error')
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    # 建立卷积神经网络
    def _cov_cnn(self, stateInput):                                                                                                                                                                                                                                           
        # first convolution kernel:8*8*4*32
        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])

        # second convolution kernel4*4*32*64:
        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])

        #third convolution kernel:3*3*64*64
        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        #full connected layer:1600*512
        W_fc1 = self.weight_variable([1600, 512])
        b_fc1 = self.bias_variable([512])

        #output layer:512*actions
        W_fc2 = self.weight_variable([512, self.n_actions])
        b_fc2 = self.bias_variable([self.n_actions])

        # combine all layers together
        # hidden layers
        #stride=4,80*80*4 to 20*20*32   since the fourth dimension of first convolution kernel is 32, and the stride is 4
        h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 4) + b_conv1)
        #20*20*32 to 10*10*32
        h_pool1 = self.maxPool_2x2(h_conv1)

        #stride=2,10*10*32 to 5*5*64, since the fourth dimension of second convolution kernel is 64, and the stride is 2
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)

        #stride=1,5*5*64 to 5*5*64
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)

        #5*5*64 to 1*1600
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        #output layer，预测的q value
        QValue = tf.matmul(h_fc1, W_fc2) + b_fc2
        return QValue, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2

    # 训练过程
    def _train_qvalue(self):
        minibatch = random.sample(self.replayMemory, self.BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # 根据下一个状态的值计算q_target，也就是q real
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT: nextState_batch})
        for i in range(0, self.BATCH_SIZE):
            y_batch.append(reward_batch[i] + self.GAMMA * np.max(QValue_batch[i]))
        _, self.cost = self.session.run([self._train_op, self.loss], 
                feed_dict={
                    self.rewardInput: y_batch,
                    self.actionInput: action_batch,
                    self.stateInput: state_batch})
        # if self.timestep % 1000 == 0:
        #     self.saver.save(self.session, './saved_networks/network' + '-dqn', global_step=self.timestep+self.initial_timestep)
        if self.timestep % self.UPDATE_TIME == 0:
            self.session.run(self.copyTargetQNetworkOperation)
        self.cost_his.append(self.cost)

    def learn(self, nextObservation, action, reward, terminal):
        newState = np.append(self.currentState[:, :, 1:], nextObservation, axis=2)
        self.replayMemory.append((self.currentState, action, reward, newState, terminal))

        #control the size of replayMemory
        if len(self.replayMemory) > self.REPLAY_SIZE:
            self.replayMemory.popleft()
        if self.timestep > self.OBSERVE and self.timestep % 2 == 0:
            self._train_qvalue()
        self.currentState = newState
        self.timestep  += 1

    def choose_action(self):
        QValue = self.QValue.eval(feed_dict={self.stateInput: [self.currentState]})[0]
        action = np.zeros(self.n_actions)

        #epsilon stragety
        l = random.random()
        if l <= self.epsilon:
            action_index = random.randrange(self.n_actions)
            action[action_index] = 1
        else:
            # print(QValue)
            action_index = np.argmax(QValue)
            action[action_index] = 1

        # change episilon
        # if self.epsilon > self.FINAL_EPSILON and self.accual_timestep > self.OBSERVE:
        #     self.epsilon= self.INITIAL_EPSILON-(self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE*self.
        # accual_timestep
        # print(self.epsilon)
        return action

    def set_init_state(self, observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis=2)

    def weight_variable(self, shape):
        # 截断的正态分布的
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def maxPool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.savefig('./cost.png')
        # plt.show()
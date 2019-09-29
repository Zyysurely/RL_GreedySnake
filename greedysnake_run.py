from snake_env import SnakeEnv
from RL_brain import DeepQNetwork
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 降低画面复杂度的预处理
def preProcess(observation):

    #将512*288的画面裁剪为80*80并将RGB(三通道)画面转换成灰度图(一通道)
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    
    # plt.imsave("res1.jpg",observation)                                   #返回(80,80,1)，最后一维是保证图像是一个tensor(张量),用于输入tensorflow
    # #将非黑色的像素都变成白色
    # threshold,observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    # plt.imsave("res.jpg",observation)                                   #返回(80,80,1)，最后一维是保证图像是一个tensor(张量),用于输入tensorflow
    return np.reshape(observation, (80, 80, 1))

def run_snake():
    brain = DeepQNetwork(4, "")
    snakeGame = SnakeEnv()
    #先随便给一个决策输入，启动游戏
    observation0, reward0, terminal,score =snakeGame.step(np.array([0, 0, 0, 1]))
    observation0 = preProcess(observation0)
    brain.setInitState(observation0[:,:,0])

    #开始正式游戏
    i = 1
    while i<=1000000:
        i = i + 1
        action = brain.choose_action()
        next_observation, reward, terminal, score = snakeGame.step(action)
        # print(reward)
        
        next_observation = preProcess(next_observation)
        brain.learn(next_observation, action, reward, terminal)
        if(i%100) == 0:
            print(i)
    
    # 画loss和round step的曲线
    brain.plot_cost()
    snakeGame.plot_cost()

if __name__ == "__main__":
    run_snake()
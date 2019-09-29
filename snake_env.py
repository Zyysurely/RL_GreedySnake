import pygame
from pygame.locals import *
from sys import exit
from pygame.color import THECOLORS
import random
import sys
import time
import math
import numpy as np


# 相关图像参数
cellsize = 20
width = 200
height = 200
cellw = int(width/cellsize)
cellh = int(height/cellsize)
speed = 15
RIGHT = 'right'
LEFT = 'left'
UP = 'up'
DOWN = 'down'
rctwidth = 0
rctwidth2 = 1
color = 0,255,0
color3 = 0,250,0
color1 = 199, 21, 133

class SnakeEnv(object):
    def __init__(self):
        self.startx = [0, 0]
        self.starty = [0, 0]
        self.snake = [[2, 0], [1, 0], [0, 0]]
        self.food = [2, 2]
        self.count = 3
        self.done = False     # 撞墙或者无法继续下去，则
        self.on_direction = RIGHT
        self.conflict = False
        self.eaten = False
        self.survive_step = 0
        self.survive_step_his = []
        pygame.init()
        self.speedCLOCK = pygame.time.Clock()
        self.screen = pygame.display.set_mode((width, height))
        self.screen.fill((0, 0, 0))
        pygame.display.flip()        
    
    def render(self):
        pass

    def reset(self):
        pass

    def reward_esitimate(self):
        dis = math.sqrt(pow((self.food[0]-self.snake[0][0]),2)+pow((self.food[1]-self.snake[0][1]),2))
        reward = (1/dis)*0.5
        return reward
    
    def step(self, action):
        self.survive_step += 1
        direction = RIGHT
        if(action[0] == 1):
            direction = RIGHT
        if(action[1] == 1):
            direction = LEFT
        if(action[2] == 1):
            direction = UP
        if(action[3] == 1):
            direction = DOWN
        self.movdirection(direction)

        # 撞墙了
        if self.snake[0][0] < 0 or self.snake[0][0] == cellw :
            self.done = True
        if self.snake[0][1] < 0 or self.snake[0][1] ==cellh :
            self.done = True
        k = self.count-1
        # 咬到尾巴了
        if self.count > 1 and self.snake[0][0] == self.snake[k][0] and self.snake[0][1] == self.snake[k][1]:
            self.done = True
        if(self.done):
            self.survive_step_his.append(self.survive_step)
            self.reward = -1
            # print("reset")
            self.startx = [0, 0]
            self.starty = [0, 0]
            self.snake = [[2, 0], [1, 0], [0, 0]]
            self.food = [2, 2]
            self.count = 3
            self.done = False
            self.on_direction = RIGHT
            self.survive_step = 0
        else:
            if(self.eaten == False):
                self.reward = 0.1 + self.reward_esitimate()
            else:
                self.food = self.randomplace()
                self.eaten = False
            if(self.conflict): # 方向不对
                self.reward = 0
                self.conflict = False
        self.drawsnake()
        img_data = pygame.surfarray.array3d(pygame.display.get_surface())
        return img_data, self.reward, self.done, self.count

    def food_in_body(self):
        for item in self.snake:
            if(item[0] == self.food[0] and item[1] == self.food[1]):
                return True
        return False     

    def movdirection(self, direction):
        # 方向相反不能生效
        # print(self.snake)
        if((direction==RIGHT and self.on_direction==LEFT)or(direction==LEFT and self.on_direction==RIGHT)or(direction==UP and self.on_direction==DOWN)\
            or (direction==DOWN and self.on_direction==UP)):
            self.conflict = True
            return
        self.on_direction = direction
        if direction == RIGHT:
            newsnkehead = [self.snake[0][0] + 1, self.snake[0][1]]
        if direction == LEFT:
            newsnkehead = [self.snake[0][0] - 1, self.snake[0][1]]
        if direction == UP :
            newsnkehead = [self.snake[0][0], self.snake[0][1] - 1]
        if direction == DOWN :
            newsnkehead = [self.snake[0][0], self.snake[0][1] + 1]
        self.snake.insert(0, newsnkehead)
        if self.food_in_body():
            self.count +=1
            self.reward = 1
            self.eaten = True
        else:
            self.snake.pop(self.count)

    def drawsnake(self):
        self.screen = pygame.display.set_mode((width, height))
        index = 0
        for coord in self.snake:
            pos = coord[0] * cellsize, coord[1] * cellsize, cellsize, cellsize
            if(index == 0):
                pos =  coord[0] * cellsize+10, coord[1] * cellsize+10
                pygame.draw.circle(self.screen, color1, pos, 10, rctwidth)
                # pygame.draw.circle(self.screen, color1, pos, 10, rctwidth2)
            else:
                pygame.draw.rect(self.screen, color3, pos, rctwidth)
                pygame.draw.rect(self.screen, color3, pos, rctwidth2)
            index += 1
            pos = self.food[0] * cellsize+10, self.food[1] * cellsize+10
            pygame.draw.circle(self.screen, (255, 255, 255), pos, 10, rctwidth)
        pygame.display.flip()
        self.speedCLOCK.tick(speed)

    def randomplace(self):
        return [random.randint(0, cellw - 1), random.randint(0, cellh - 1)]
    
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.figure()
        print(self.survive_step)
        plt.plot(np.arange(len(self.survive_step_his)), self.survive_step_his)
        plt.ylabel('survive steps')
        plt.xlabel('rounds')
        plt.savefig('./t2.png')
        # plt.show()

if(__name__ == "__main__"):
    snake = SnakeEnv()
    epsilon = 0
    while epsilon <= 7:
       snake.step()
       epsilon += 1
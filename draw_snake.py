import pygame
from pygame.locals import *
from sys import exit
from pygame.color import THECOLORS
import random
import sys
cellsize = 20
width = 640
height = 480
cellw = int(width/cellsize)
cellh = int(height/cellsize)
speed = 15
right = 'right'
left = 'left'
up = 'up'
down = 'down'
rctwidth = 0
rctwidth2 = 1
color = 0,255,0
color3 = 0,250,0
direction = right
ap = { 'x': 0, 'y': 0}
color1 = 199, 21, 133
count = 3

def main():
    global speed, speedCLOCK,ap,count, screen
    pygame.init()
    count = 3
    speedCLOCK = pygame.time.Clock()
    pygame.display.set_caption("Hello, World!")
    screen = pygame.display.set_mode((width, height))
    screen.fill((0, 0, 0))
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            keys = pygame.key.get_pressed()
            if keys[K_BACKSPACE]:
                pygame.quit()
                sys.exit()
        ap = randomplace()
        snakeoriginalbody()
        gameover()
        print(randomplace())
        return

def snakeoriginalbody():
    k = 10
    startx = random.randint( 5, cellw - 6)
    starty = random.randint( 5, cellh - 6)
    snake = [ { 'x': startx, 'y': starty},
              { 'x': startx - 1, 'y': starty },
              { 'x': startx - 2, 'y': starty }]
    while True:
        movdirection( snake)
        drawsnake(snake)
        pygame.display.update()
        if snake[0]['x'] < 0 or snake[0]['x'] == cellw :
            return
        if snake[0]['y'] < 0 or snake[0]['y'] ==cellh :
            return
        k = count-1
        while k >1 :
            if snake[0]['x'] == snake[k]['x'] and snake[0]['y'] == snake[k]['y']:
                return
            k -= 1

def movdirection( snake):
    global direction,ap,count
    keys = pygame.key.get_pressed()
    if keys[K_RIGHT]:
        direction = right
    if keys[K_LEFT]:
        direction = left
    if keys[K_UP]:
        direction = up
    if keys[K_DOWN]:
        direction = down
    if keys[K_BACKSPACE]:
        pygame.quit()
        sys.exit()
    if direction == right :
        newsnkehead = { 'x': snake[0]['x'] + 1, 'y': snake[0]['y']}
    if direction == left :
        newsnkehead = { 'x': snake[0]['x'] - 1, 'y': snake[0]['y']}
    if direction == up :
        newsnkehead = { 'x': snake[0]['x'], 'y': snake[0]['y'] - 1}
    if direction == down :
        newsnkehead = { 'x': snake[0]['x'], 'y': snake[0]['y'] + 1}
    snake.insert(0, newsnkehead)
    if snake[0]['x'] == ap['x'] and snake[0]['y'] == ap['y']:
        ap = randomplace()
        count +=1
        return
    else:
        snake.pop(count)

def drawsnake(snake):
    global screen
    screen = pygame.display.set_mode((width, height))
    for coord in snake:
            pos = coord['x'] * cellsize, coord['y'] * cellsize, cellsize, cellsize
            pygame.draw.rect(screen, color, pos, rctwidth)
            pygame.draw.rect(screen, color3, pos, rctwidth2)
            pos = ap['x'] * cellsize, ap['y'] * cellsize,cellsize,cellsize
            pygame.draw.rect(screen, color1, pos, rctwidth)  
            over = pygame.font.SysFont('arial', 60)
            textImage = over.render("socre %.d" % (count - 3), True, color)
            screen.blit(textImage, (450, 60))
    over = pygame.font.SysFont('arial', 60)
    textImage = over.render("socre %.d" % (count - 3), True, color)
    screen.blit(textImage, (450, 60))
    speedCLOCK.tick(speed)
    pygame.display.flip()

def randomplace():
    return{ 'x': random.randint( 0, cellw - 1), 'y': random.randint( 0, cellh - 1)}
def gameover():
    screen = pygame.display.set_mode((width, height))
    over = pygame.font.SysFont('arial', 60)
    over2 = pygame.font.SysFont('arial',30)
    textImage = over.render("game over", True, color)
    screen.blit(textImage, (200, 200))
    textImage2 = over2.render("press L-CTRL to restar", True, color)
    screen.blit(textImage2, (200, 300))
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            keys = pygame.key.get_pressed()
            if  keys[K_BACKSPACE]:
                pygame.quit()
                sys.exit()
            if  keys[K_LCTRL]:
                main()
main()
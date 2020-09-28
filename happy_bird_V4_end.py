#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pygame.locals import KEYDOWN, K_ESCAPE, K_q ,K_x, K_w, K_v, K_f, FULLSCREEN
import pygame
from pygame.locals import Color, QUIT, MOUSEBUTTONDOWN, USEREVENT, USEREVENT
import cv2
import sys
# import tensorflow as tf
import numpy as np
import time
import random

import os
import sqlite3
from datetime import datetime,timezone,timedelta

# 建立database
db = sqlite3.connect('HAPPY_BIRD/game.db')
cursor = db.cursor()

# 取目前USER
with open('HAPPY_BIRD/User.txt','r') as f:
    test = f.read().splitlines()
User = test[0]

# 視窗大小
WIN_WIDTH = 1280
WIN_HEIGHT = 720

# 鳥鳥大小
IMAGEWIDTH = 50
IMAGEHEIGHT = 50

# 按鈕大小
BUTTONWIDTH = 256
BUTTONHEIGHT = 128

# 開始結束畫面
# STAT_FONT = pygame.font.SysFont("comicsans", 50)
# END_FONT = pygame.font.SysFont("comicsans", 70)



face_cascade = cv2.CascadeClassifier('HAPPY_BIRD/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('HAPPY_BIRD/haarcascade_smile.xml') 





def detect_smile(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces: 
        # cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2) 
        roi_gray = gray[y:y + h, x:x + w] 
        roi_color = frame[y:y + h, x:x + w] 
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        # for (sx, sy, sw, sh) in smiles: 
            # cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
        
        if len(smiles):
            return 1
        else:
            return 0





class Pipe():
    """
    represents a pipe object
    """
    GAP = 250
    VEL = 10

    def __init__(self, x):
        """
        initialize pipe object
        :param x: int
        :param y: int
        :return" None
        """
        self.x = x
        self.height = 0

        # where the top and bottom of the pipe is
        self.top = 0
        self.bottom = 0
        
        pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("HAPPY_BIRD/imgs","pipe.png")))
        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        self.PIPE_BOTTOM = pipe_img
        
#         self.rect = self.image.get_rect()
#         self.rect.topleft = (self.x, self.y)

        self.passed = False

        self.set_height()

    def set_height(self):
        """
        set the height of the pipe, from the top of the screen
        :return: None
        """
        self.height = random.randrange(25, 300)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        """
        move pipe based on vel
        :return: None
        """
        self.x -= self.VEL

    def draw(self, win):
        """
        draw both the top and bottom of the pipe
        :param win: pygame window/surface
        :return: None
        """
        # draw top
        win.blit(self.PIPE_TOP, (self.x, self.top))
        # draw bottom
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))
       
        
      
        
        
bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("HAPPY_BIRD/imgs","bird" + str(x) + ".png"))) for x in range(1,4)]        


class Bird:
    """
    Bird class representing the flappy bird
    """
    MAX_ROTATION = 25
    IMGS = bird_images
    ROT_VEL = 20
    ANIMATION_TIME = 5
    VEL_drop = 12
    VEL_jump = 26 # 13

    def __init__(self, x, y):
        """
        Initialize the object
        :param x: starting x pos (int)
        :param y: starting y pos (int)
        :return: None
        """
        self.x = x
        self.y = y
        self.img_width = IMAGEWIDTH
        self.img_height = IMAGEHEIGHT
        self.tilt = 0  # degrees to tilt
        self.tick_count = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]
#         self.rect = self.image.get_rect()
        
    def draw(self, win):
        """
        draw the bird
        :param win: pygame window or surface
        :return: None
        """
        self.img_count += 1

        # For animation of bird, loop through three images
        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        # so when bird is nose diving it isn't flapping
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2


        # tilt the bird
        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)

        # win.blit(self.IMGS, (self.x, self.y))
    
    
    def drop(self):
        self.y += self.VEL_drop
        
    def jump(self):
        self.y -= self.VEL_jump

        
def blitRotateCenter(surf, image, topleft, angle):
    """
    Rotate a surface and blit it to the window
    :param surf: the surface to blit to
    :param image: the image surface to rotate
    :param topLeft: the top left position of the image
    :param angle: a float value for angle
    :return: None
    """
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)

    surf.blit(rotated_image, new_rect.topleft)

def check_collision(pipes):
    for pipe in pipes:
        if bird_rect.colliderect(pipe):
            return False

    if bird_rect.top <= -100 or bird_rect.bottom >= 900:
        return False

    return True


def update_score(score, high_score):
    if score > high_score:
        high_score = score
    return high_score


# 時區轉換
def TW_Time(T):
    dt1 = datetime.strptime(T, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    return dt2.strftime("%Y-%m-%d %H:%M:%S")
################################################################################################
class Button(pygame.sprite.Sprite):
    def __init__(self, x, y, picture):
        super().__init__()
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.raw_image = pygame.image.load(picture)
        self.image = pygame.transform.scale(self.raw_image, (BUTTONWIDTH, BUTTONHEIGHT))
        self.rect = self.image.get_rect()
        self.rect.center = (self.x, self.y)
        self.width = BUTTONWIDTH
        self.height = BUTTONHEIGHT
        self.widow_width = WIN_WIDTH
        self.window_height = WIN_HEIGHT
        
    def draw(self, win):
        win.blit(self.image, self.rect)
################################################################################################
button_start = Button(640, 360, 'HAPPY_BIRD/imgs/start.png')
button_exit = Button(1100, 650, 'HAPPY_BIRD/imgs/exit.png')
button_demo = Button(290, 360, 'HAPPY_BIRD/imgs/demo.png')
button_score = Button(990, 360, 'HAPPY_BIRD/imgs/score.png')
button_menu = Button(180, 650, 'HAPPY_BIRD/imgs/menu.png')
button_again = Button(640, 650, 'HAPPY_BIRD/imgs/again.png')
############################################
camera = cv2.VideoCapture(0)
camera_x, camera_y = (WIN_WIDTH, WIN_HEIGHT)     # Set camera resolution [1]=1280,720
camera.set(cv2.CAP_PROP_FRAME_WIDTH , camera_x)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_y)

# 起始背景
bg = pygame.transform.scale(pygame.image.load('HAPPY_BIRD/imgs/bg.png'),(WIN_WIDTH, WIN_HEIGHT))
# bg = pygame.image.load('imgs/bg.png')
bg_score = pygame.image.load('HAPPY_BIRD/imgs/bg_score.png')



pygame.init()
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
game_font = pygame.font.Font('HAPPY_BIRD/04B_19.ttf',40)
# Game Variables
gravity = 0.25
bird_movement = 0
score = 0
high_score = 0


def main():
    

    
    try:
        pygame.init()
        WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption("Flappy Bird")
        pipe_move_event = USEREVENT+1
        pygame.time.set_timer(pipe_move_event, 100) # 設定每300毫秒更新一次

        # 導入全螢幕設定
        Fullscreen = False

                    
        pipe1 = Pipe(WIN_WIDTH)
        pipe2 = Pipe(WIN_WIDTH/2)
        bird = Bird(10,240)
       
# WIN_WIDTH = 1280
# WIN_HEIGHT = 720  

        game_start_surface = pygame.transform.scale2x(pygame.image.load('HAPPY_BIRD/assets/sprites/message.png').convert_alpha())
        game_start_rect = game_start_surface.get_rect(center = (WIN_WIDTH/2,WIN_HEIGHT/2))
        game_over_surface = pygame.transform.scale2x(pygame.image.load('HAPPY_BIRD/assets/sprites/gameover.png').convert_alpha())
        game_over_rect = game_over_surface.get_rect(center = (WIN_WIDTH/2,50))
        score_sound_countdown = 100
        clock = pygame.time.Clock()
        high_score = 0
        score = 0  

        
        
        run = True   

        while run:
            Menu = True
            while Menu:

                # 加上背景圖片
                WIN.blit(bg, (0, 0))
                
                # 加上按鈕
                button_start.draw(WIN)
                button_exit.draw(WIN)
                button_demo.draw(WIN)
                button_score.draw(WIN)
                
                # 滑鼠位置偵測
                mouse_pos = pygame.mouse.get_pos()
                 #事件迴圈
                for event in pygame.event.get():

                    # 關閉事件 (按下右上叉叉，離開遊戲)
                    if event.type == pygame.QUIT:
                        On_status = False
                        sys.exit(0)
                        camera.release()

                    # 鍵盤事件
                    elif event.type == KEYDOWN:
                        
                         # 按下ESC or v，離開遊戲
                        if event.key == K_ESCAPE or event.key == K_v:
                            On_status = False
                            sys.exit(0)
                            camera.release()
                        
#                         # 按下x，開始遊戲
#                         elif event.key == K_x:
#                             Menu = False
#                             Run = True
#                             Score = False
#                             Demo = False
                            
#                         # 按下z，開始Demo    
#                         elif event.key == K_z:
#                             Menu = False
#                             Run = False
#                             Score = False
#                             Demo = True
                        
#                         # 按下c，積分頁面
#                         elif event.key == K_c:
#                             Menu = False
#                             Run = False
#                             Score = True
#                             Demo = False
                        
                        # 按下F，切換全螢幕
                        elif event.key == K_f:
                            Fullscreen = not Fullscreen
                            if Fullscreen:
                                screen = pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                            else:
                                screen = pygame.display.set_mode((1280, 720), 0, 32)
                    
                    # 滑鼠事件
                    elif event.type == MOUSEBUTTONDOWN:
                        
                        # 按下start按鈕，開始遊戲
                        if button_start.rect.collidepoint(mouse_pos):
                            Menu = False
                            start = True
                            stop = False
                            Demo = False

                        # 按下score按鈕，積分頁面
                        elif button_score.rect.collidepoint(mouse_pos):
                            Menu = False
                            start = False
                            running = False
                            stop = True
                            Demo = False  
                            
                        # 按下demo按鈕，開始Demo    
                        elif button_demo.rect.collidepoint(mouse_pos):
                            Menu = False
                            start = False
                            running = False
                            stop = False
                            Demo = True                           
                        
                        # 按下exit按鈕，離開遊戲
                        elif button_exit.rect.collidepoint(mouse_pos):
                            On_status = False
                            sys.exit(0)
                            camera.release()
                            
                # 循環更新畫面
                pygame.display.update()

#             start = True
            while start:
                ret, frame = camera.read()
                WIN.fill([0, 0, 0])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 1) # 左右對調
                sm = detect_smile(frame)
                frame = frame.swapaxes(0, 1)
                pygame.surfarray.blit_array(WIN, frame)
                WIN.blit(game_start_surface,game_start_rect)
                
                for event in pygame.event.get():                
                    if event.type == KEYDOWN:
                        if event.key == K_ESCAPE or event.key == K_q:
                            sys.exit(0)
                            camera.release()
                        elif event.key == K_w:
                            running = True
                            start  = False
                            score = 0

                        # 按下F，切換全螢幕
                        elif event.key == K_f:
                            Fullscreen = not Fullscreen
                            if Fullscreen:
                                screen = pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                            else:
                                screen = pygame.display.set_mode((1280, 720), 0, 32)
                
                
                pygame.display.update()
                
            while running:

                ret, frame = camera.read()
                WIN.fill([0, 0, 0])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 1) # 左右對調
                sm = detect_smile(frame)
                frame = frame.swapaxes(0, 1)
                pygame.surfarray.blit_array(WIN, frame)

                pipe1.draw(WIN)
                pipe2.draw(WIN)
                bird.draw(WIN)

                for event in pygame.event.get():


                    if event.type == pygame.QUIT:
                        sys.exit(0)
                        camera.release()

                    if event.type == KEYDOWN:
                        if event.key == K_x:
                            bird.jump()

                        elif event.key == K_q:

                            sys.exit(0)
                            camera.release()

                        # 按下F，切換全螢幕
                        elif event.key == K_f:
                            Fullscreen = not Fullscreen
                            if Fullscreen:
                                screen = pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                            else:
                                screen = pygame.display.set_mode((1280, 720), 0, 32)

                    if event.type == pipe_move_event:
                        pipe1.move()
                        pipe2.move()
                        bird.drop()

                        # 水管輪動
                        if pipe1.x==0:
                            pipe1 = Pipe(WIN_WIDTH)                      
                            pipe1.draw(WIN)                        
                        if pipe2.x==0:
                            pipe2 = Pipe(WIN_WIDTH)
                            pipe2.draw(WIN)
                            



                        # 水管碰撞 

                        if pipe1.x == bird.x + bird.img_width:
                            if pipe1.height >= bird.y:
                                
                                print('pipe1_height:', pipe1.height)
                                pipe1 = Pipe(WIN_WIDTH)
                                pipe2 = Pipe(WIN_WIDTH/2)
                                bird = Bird(10,240)
                                running = False
                                stop = True
                                
                            elif pipe1.bottom <= (bird.y + bird.img_height):
                               
                                print('pipe1_bottom:', pipe1.bottom)
                                pipe1 = Pipe(WIN_WIDTH)
                                pipe2 = Pipe(WIN_WIDTH/2)
                                bird = Bird(10,240)
                                running = False
                                stop = True
                            else:
                                score += 1
                                print(score)



                        if pipe2.x == bird.x + bird.img_width:
                            if pipe2.height >= bird.y:                           
                                print('pipe2_height:', pipe2.height)
                                pipe1 = Pipe(WIN_WIDTH)
                                pipe2 = Pipe(WIN_WIDTH/2)
                                bird = Bird(10,240)
                                running = False
                                stop = True
                            elif pipe2.bottom <= (bird.y + bird.img_height):
                                print('pipe2_bottom:', pipe2.bottom)
                                pipe1 = Pipe(WIN_WIDTH)
                                pipe2 = Pipe(WIN_WIDTH/2)
                                bird = Bird(10,240)
                                running = False
                                stop = True
                            else:
                                score += 1
                                print(score)


                        # 上下牆壁
                        if bird.y + bird.img_height >= WIN_HEIGHT - 5 or bird.y - 10 <= 0:
                            pygame.time.delay(10)
                            bird = Bird(10,240)
                            running = False
                            stop = True
                            print('touch padding')


                        # 水管加分
    #                     if game
                        # 微笑動作
                        if sm==1 :
                            bird.jump()


                        if running == False:
                            clock 

                head_font = pygame.font.Font('HAPPY_BIRD/04B_19.ttf',40)                
                score_surface = head_font.render(f'Score: {int(score)}', True, (255, 255, 255))
                
    #             highscore_surface = head_font.render(f'HighScore: {int(high_score)}', True, (255, 255, 255))
                WIN.blit(score_surface, (80,100))  
    #             WIN.blit(highscore_surface, (80,50))
                cursor.execute('INSERT INTO `happy_bird`( `name`, `score`) VALUES (?,?)',[User,score])
                db.commit()





                pygame.display.update()

            print(score)

    #         if game_active:
    #             score_display('main_game')
    #         else:
    #             WIN.blit(game_over_surface,game_over_rect)
    #             high_score = update_score(score,high_score)
    #             score_display('game_over')


#             stop = True
            while stop :

                for event in pygame.event.get():
#                     pygame.init()
                    # 建立 window 視窗畫布，大小為 800x600
#                     window_surface = pygame.display.set_mode((800, 600))
                    # 設置視窗標題為 Hello World:)
#                     pygame.display.set_caption('Hello World:)')
                    # 清除畫面並填滿背景色
                    WIN.blit(bg, (0, 0))

                    # 宣告 font 文字物件
                    head_font = pygame.font.Font('HAPPY_BIRD/chinese.msyh.ttf',60) 
                    # 渲染方法會回傳 surface 物件
                    WIN.blit(game_over_surface,game_over_rect)
                    text_surface = head_font.render(f'Score: {int(score)}', True, (0, 0, 0))
                    
                    
                    CS = list(cursor.execute('SELECT * FROM `happy_bird` WHERE `name` = ? Order BY `time` DESC',[User]))
                    if len(CS):
                        current_score = head_font.render(f'目前分數:   {CS[0][2]}分   {TW_Time(CS[0][3])[:10]}', True, (0, 255, 0))
                        WIN.blit(current_score, (200, 150))

                    HS = list(cursor.execute('SELECT * FROM `happy_bird` WHERE `name` = ? Order BY `score` DESC',[User]))
                    if len(CS):
                        highest_score = head_font.render(f'最高分數:   {HS[0][2]}分   {TW_Time(HS[0][3])[:10]}', True, (255, 127, 80))
                        WIN.blit(highest_score, (200, 250))

#                     if len(CS) >= 4:
#                         History = head_font.render('歷史分數', True, (30, 144, 255))
#                         WIN.blit(History, (200, 230)+20)
#                         for i in range(1,4):
#                             History_score = head_font.render(f'前 {i} 次分數:   {CS[i][2]}分   {TW_Time(CS[i][3])[:10]}', True, (30, 144, 255))
#                             WIN.blit(History_score, (200, 230+80*i))
#                     elif len(CS) == 3:
#                         History = head_font.render('歷史分數', True, (30, 144, 255))
#                         WIN.blit(History, (200, 230))
#                         for i in range(1,3):
#                             History_score = head_font.render(f'前 {i} 次分數:   {CS[i][2]}分   {TW_Time(CS[i][3])[:10]}', True, (30, 144, 255))
#                             WIN.blit(History_score, (200, 230+80*i))
#                     elif len(CS) == 2:
#                         History = head_font.render('歷史分數', True, (30, 144, 255))
#                         WIN.blit(History, (200, 230))
#                         for i in range(1,2):
#                             History_score = head_font.render(f'前 {i} 次分數:   {CS[i][2]}分   {TW_Time(CS[i][3])[:10]}', True, (30, 144, 255))
#                             WIN.blit(History_score, (200, 230+80*i))
                    # blit 用來把其他元素渲染到另外一個 surface 上，這邊是 window 視窗
#                     text_surface = head_font.render(f'Score: {int(score)}', True, (0, 0, 0))
#                     WIN.blit(text_surface, (WIN_WIDTH/2,50))
                    

                    # 更新畫面，等所有操作完成後一次更新（若沒更新，則元素不會出現）
                    
#                     running = True
#                     pygame.display.update()
                    
                            
                    button_menu.draw(WIN)       
                    mouse_pos = pygame.mouse.get_pos()
                    
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()

                    elif event.type == KEYDOWN:
                        if event.key == K_ESCAPE or event.key == K_q:
                            sys.exit(0)
                            camera.release()
                        if event.key == K_w:
                            start = True
                            running = True
                            stop = False
                            score = 0

                        # 按下F，切換全螢幕
                        elif event.key == K_f:
                            Fullscreen = not Fullscreen
                            if Fullscreen:
                                screen = pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                            else:
                                screen = pygame.display.set_mode((1280, 720), 0, 32)
                            
                    elif event.type == MOUSEBUTTONDOWN:
                        if button_menu.rect.collidepoint(mouse_pos):
                            start = True
                            running = True
                            stop = False
                            score = 0 
                    
                pygame.display.update() 
                
                # 進入demo
#############################################################
            demo_video = cv2.VideoCapture('HAPPY_BIRD/imgs/DEMO_HAPPY_BIRD.mp4')
            while Demo:
                
                hasFrame, img = demo_video.read()
                time.sleep(0.001)    
                if not hasFrame:
                    Menu = True
                    start = False
                    stop = False
                    Demo = False
                    break
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.swapaxes(0, 1)
        
        
                pygame.surfarray.blit_array(WIN, img)        
                button_menu.draw(WIN)       
                mouse_pos = pygame.mouse.get_pos()
                

                for event in pygame.event.get():

                    # 關閉事件1 (按下右上叉叉，離開遊戲)
                    if event.type == pygame.QUIT:
                        On_status = False
                        sys.exit(0)
                        camera.release()

                    # 關閉事件2 (按下ESC 或 Q，離開遊戲)
                    elif event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            On_status = False
                            sys.exit(0)
                            camera.release()

                        elif event.key == K_x:
                            Menu = True
                            Run = False
                            Score = False
                            Demo = False

                        # 按下F，切換全螢幕
                        elif event.key == K_f:
                            Fullscreen = not Fullscreen
                            if Fullscreen:
                                screen = pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                            else:
                                screen = pygame.display.set_mode((1280, 720), 0, 32)
                        
                    elif event.type == MOUSEBUTTONDOWN:
                        if button_menu.rect.collidepoint(mouse_pos):
                            Menu = True
                            start = False                           
                            stop = False
                            Demo = False                          

                pygame.display.update()

#############################################################

                            

            

                        


    except (KeyboardInterrupt, SystemExit):
        pygame.quit()
        cv2.destroyAllWindows()
        camera.release()

        
if __name__ == '__main__':    
    main()


# In[ ]:


# class Mosquito(pygame.sprite.Sprite):
#     def init(self):
#         super().init()
#         pygame.sprite.Sprite.init(self)
#         self.x = random.randrange(0, 640-IMAGEWIDTH)
#         self.y = random.randrange(-20, 0)
#         self.raw_image = pygame.image.load('UFO.png')
#         self.image = pygame.transform.scale(self.raw_image, (IMAGEWIDTH, IMAGEHEIGHT))
#         self.rect = self.image.get_rect()
#         self.rect.topleft = (self.x, self.y)
#         self.width = IMAGEWIDTH
#         self.height = IMAGEHEIGHT
#         self.widow_width = WINDOW_WIDTH
#         self.window_height = WINDOW_HEIGHT

#     def draw(self, win):
#         win.blit(self.image, self.rect)
#     def move(self):
#         self.x += random.randrange(-10, 10)
#         self.y += 10
#         self.rect.topleft = (self.x, self.y)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


# tensorflow 2.3
# cv2.VideoCapture(0) 不一樣的話記得更換

from pygame.locals import MOUSEBUTTONDOWN ,KEYDOWN, K_ESCAPE, K_q, K_x, K_f, K_z, K_c, K_v, FULLSCREEN
import pygame
from pygame.locals import Color, QUIT, MOUSEBUTTONDOWN, USEREVENT, USEREVENT
import cv2
import sys
import tensorflow as tf
import numpy as np
import time
import random
import sqlite3
from datetime import datetime,timezone,timedelta

# 連結Database
db = sqlite3.connect('game.db')
cursor = db.cursor()

# 讀取User資訊
with open('User.txt','r') as f:
    test = f.read().splitlines()
User = test[0]

# 按鈕大小
BUTTONWIDTH = 256
BUTTONHEIGHT = 128

# 視窗大小
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# 圖片大小
IMAGEWIDTH = 250
IMAGEHEIGHT = 200

# 時區轉換
###############################################################################################
def TW_Time(T):
    dt1 = datetime.strptime(T, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    return dt2.strftime("%Y-%m-%d %H:%M:%S")
#################################################################################################

# tf姿勢預測部分
####################################################################################################
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def parse_output(heatmap_data,offset_data):
    joint_num = heatmap_data.shape[-1]
    pose_kps = np.zeros((joint_num,3),np.uint8)
    for i in range(heatmap_data.shape[-1]):
        joint_heatmap = heatmap_data[...,i]
        max_val_pos = np.squeeze(np.argwhere(joint_heatmap==np.max(joint_heatmap)))
        remap_pos = np.array(max_val_pos/8*257,dtype=np.int32)
        pose_kps[i,0] = int((remap_pos[0] + offset_data[max_val_pos[0],max_val_pos[1],i]))
        pose_kps[i,1] = int((remap_pos[1] + offset_data[max_val_pos[0],max_val_pos[1],i+joint_num]))
        pose_kps[i,2] = (sigmoid(heatmap_data[:,:,i]).sum())*100/3
    return pose_kps



model = tf.lite.Interpreter('NINJA_WAR/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite')
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()
###################################################################################################

# 雷切部分
##################################################################################################
# 隨機位置(雷電生成的位置)
def get_random_position(widow_width, window_height, image_width, image_height):
    random_x = random.randint(0, widow_width-IMAGEWIDTH)
    random_y = random.randint(0, window_height-IMAGEHEIGHT)
    return random_x, random_y

# 雷電參數
class Light(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        pygame.sprite.Sprite.__init__(self)
        self.x = random.randrange(0,1280-IMAGEWIDTH)
        self.y = random.randrange(0,480-IMAGEHEIGHT)
        self.raw_image = pygame.image.load('NINJA_WAR/light.png')
        self.image = pygame.transform.scale(self.raw_image, (IMAGEWIDTH, IMAGEHEIGHT))
        self.rect = self.image.get_rect()
        self.rect.topleft = (self.x, self.y)
        self.width = IMAGEWIDTH
        self.height = IMAGEHEIGHT
        self.widow_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
        
        
    def draw(self, win):
        win.blit(self.image, self.rect)
    def move(self):
        self.x += random.randrange(-15, 15)
        self.y += random.randrange(-15, 15)
        self.rect.topleft = (self.x, self.y)

###############################################################################################
   
    
    
# 螺旋丸部分    
###############################################################################################
class Sprial(pygame.sprite.Sprite):
    def __init__(self,  x, y):
        super().__init__()
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.raw_image = pygame.image.load('NINJA_WAR/sprial.png')     
        self.image = pygame.transform.scale(self.raw_image, (IMAGEWIDTH, IMAGEHEIGHT))
        self.rect = self.image.get_rect()
        self.rect.center = (self.x, self.y)
        self.width = IMAGEWIDTH
        self.height = IMAGEHEIGHT
        self.widow_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
###############################################################################################
sprial = Sprial(0, 0)
sprial2 = Sprial(0, 0)


# 按鈕部分
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
        self.widow_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
        
    def draw(self, win):
        win.blit(self.image, self.rect)
################################################################################################
button_start = Button(640, 360, 'NINJA_WAR/button/start.png')
button_exit = Button(1100, 650, 'NINJA_WAR/button/exit.png')
button_demo = Button(290, 360, 'NINJA_WAR/button/demo.png')
button_score = Button(990, 360, 'NINJA_WAR/button/score.png')
button_menu = Button(180, 650, 'NINJA_WAR/button/menu.png')
button_again = Button(640, 650, 'NINJA_WAR/button/again.png')



# 鏡頭選擇
camera = cv2.VideoCapture(0)
camera_x, camera_y = (1280, 720)
camera.set(cv2.CAP_PROP_FRAME_WIDTH , camera_x)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_y)

# 起始背景
bg = pygame.image.load('NINJA_WAR/button/bg.jpg')
bg = pygame.transform.scale(bg, (1280, 720))
bg_score = pygame.image.load('NINJA_WAR/button/bg_score.png')


def main():
           
    try:
        
        # 初始化
        pygame.init()
        
        # 視窗畫面大小
        window_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # 遊戲名稱
        pygame.display.set_caption('Ninja War')
        
        # 遊戲icon
        logo = pygame.image.load('NINJA_WAR/shuriken.jpg')
        pygame.display.set_icon(logo)

        # 導入全螢幕設定
        Fullscreen = False
        
        # 雷電參數設定
        light = Light()
        
        # 事件設定
        # 共兩件事 
        # 1. 雷電移動 
        reload_light_event = USEREVENT+1
        pygame.time.set_timer(reload_light_event, 300) # 設定每300毫秒更新一次
        # 2. 殺死雷電
        kill_event = USEREVENT
        pygame.time.set_timer(kill_event, 100) # 設定每100毫秒更新一次
        
        # 分數起始值
        points = 0
        
        # 時間起始值
        T = 30
        
        # 各文字字形與大小設定
        my_font = pygame.font.SysFont(None, 60)
        my_hit_font = pygame.font.SysFont(None, 240)
        
        
        sprial = Sprial(0, 0)
        sprial2 = Sprial(0, 0)
        
        bg2 = pygame.image.load('NINJA_WAR/background.png').convert_alpha()
        bg2 = pygame.transform.scale(bg2, (1280, 720))
        bg2 = pygame.transform.flip(bg2 , True, False)
        
# 事件設定
#################################################################################
        # 共三件事 
        # 1. ufo移動 
        reload_ufo_event = USEREVENT+1
        pygame.time.set_timer(reload_ufo_event, 200) # 設定每300毫秒更新一次
        
        # 2. 撞到ufo
        kill_event = USEREVENT+2
        pygame.time.set_timer(kill_event, 60) # 設定每100毫秒更新一次
        
        # 3. 碰撞按鈕 (已關閉)
        select_button = USEREVENT+3
        pygame.time.set_timer(select_button, 100) # 設定每100毫秒更新一次
        
        gameover = USEREVENT+4
        pygame.time.set_timer(gameover, 100) # 設定每100毫秒更新一次
        
#################################################################################

        # 分數起始值
        points = 0
        
        # 時間起始值
        T = 30
        
        # 各文字字形與大小設定
        my_font = pygame.font.Font('chinese.msyh.ttf', 60)
        my_hit_font = pygame.font.Font('chinese.msyh.ttf', 240)
        
        
# 進入Pygame      
###############################################################################################################        
        On_status = True
        Menu = True
        while On_status:
#             bonber = Bonber(0, 0)
            
# 進入選單 
###################################################################
            
            while Menu:

                # 加上背景圖片
                window_surface.blit(bg, (0, 0))
                
                # 加上按鈕
                button_start.draw(window_surface)
                button_exit.draw(window_surface)
                button_demo.draw(window_surface)
                button_score.draw(window_surface)
                
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
                        # 按下F，切換全螢幕
                        if event.key == K_f:
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
                            Run = True
                            Score = False
                            Demo = False

                        # 按下score按鈕，積分頁面
                        elif button_score.rect.collidepoint(mouse_pos):
                            Menu = False
                            Run = False
                            Score = True
                            Demo = False  

                        # 按下demo按鈕，開始Demo    
                        elif button_demo.rect.collidepoint(mouse_pos):
                            Menu = False
                            Run = False
                            Score = False
                            Demo = True                           

                        # 按下exit按鈕，離開遊戲
                        elif button_exit.rect.collidepoint(mouse_pos):
                            On_status = False
                            sys.exit(0)
                            camera.release()

                        # 循環更新畫面
                pygame.display.update()

        
# 開始遊戲
###################################################################
  
            while Run:

                # tf姿勢預測部分
                ###################################################################################
                ret, frame = camera.read()

                window_surface.fill([0, 0, 0])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 1) # 左右對調


                input_img = tf.reshape(tf.image.resize(frame[:,:,::-1], [257,257]), [1,257,257,3])
                floating_model = input_details[0]['dtype'] == np.float32
                if floating_model:
                    input_img = (np.float32(input_img) - 127.5) / 127.5
                model.set_tensor(input_details[0]['index'], input_img)
                start = time.time()
                model.invoke()

                output_data =  model.get_tensor(output_details[0]['index'])
                offset_data = model.get_tensor(output_details[1]['index'])
                heatmaps = np.squeeze(output_data)
                offsets = np.squeeze(offset_data)

                show_img = np.squeeze((input_img.copy()*127.5+127.5)/255.0)[:,:,::-1]
                show_img = np.array(show_img*255,np.uint8)
                kps = parse_output(heatmaps,offsets)
                ####################################################################################

                # 將webcam兩軸互換 ( np.array[y,x]=>x,y )
                frame = frame.swapaxes(0, 1)

                # 正式放入webcam影像
                pygame.surfarray.blit_array(window_surface, frame)

                # tf姿勢預測結果 (這邊只放鼻子)
                hand1_Score = kps[9,2]
                hand2_Score = kps[10,2]
                hand1_x = int(kps[9,1]*1280/257)
                hand1_y = int(kps[9,0]*720/257)
                hand2_x = int(kps[10,1]*1280/257)
                hand2_y = int(kps[10,0]*720/257)

                if hand1_Score >= 25:
                        sprial = Sprial(hand1_x, hand1_y)
                        window_surface.blit(sprial.image, sprial.rect)
                if hand2_Score >= 25:
                        sprial2 = Sprial(hand2_x, hand2_y)
                        window_surface.blit(sprial2.image, sprial2.rect)


                window_surface.blit(bg2,(0,0))


                light.draw(window_surface)
                
                # 正式處理事件
                for event in pygame.event.get():
                    #計時器
                    if event.type == gameover:
                        if T <= 0:
#                             print(T)
#                             print(points)
#                             print(User)
                            cursor.execute('INSERT INTO ninja_war( name, score) VALUES (?,?)',[User,points])
                            db.commit()
                            light.kill()
                            light = Light()
                            
                            
                            points = 0
                            T = 30
                            Score = True
                            Menu = False
                            Run = False
                            
                            Demo = False
                    # 關閉事件1 (按下右上叉叉，離開遊戲)
                    elif event.type == pygame.QUIT:
                        sys.exit(0)
                        camera.release()

                    # 鍵盤事件
                    elif event.type == KEYDOWN:
                        # 按下F，切換全螢幕
                        if event.key == K_f:
                            Fullscreen = not Fullscreen
                            if Fullscreen:
                                screen = pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                            else:
                                screen = pygame.display.set_mode((1280, 720), 0, 32)

                    # 關閉事件2 (按下ESC 或 Q，離開遊戲)
                    elif event.type == KEYDOWN:
                        if event.key == K_ESCAPE or event.key == K_q:
                            sys.exit(0)
                            camera.release()

                    # 雷電移動 (我這邊是直接殺死一隻，在創造一隻，製造位移假象)
                    elif event.type == reload_light_event:
                        light.move()



                    # 殺死雷電
                    elif event.type == kill_event:
                        if pygame.sprite.collide_rect(light, sprial):
                            cursor.execute('INSERT INTO `ninja_war`( `name`, `score`) VALUES (?,?)',[User,points])
                            db.commit()
                            light.kill()
                            light = Light()

                            points += 1  # 分數累計

                        if pygame.sprite.collide_rect(light, sprial2):
                            cursor.execute('INSERT INTO `ninja_war`( `name`, `score`) VALUES (?,?)',[User,points])
                            db.commit()
                            light.kill()
                            light = Light()

                            points += 1  # 分數累計

                T -= 0.1 # 時間累計 

                # 遊戲分數與時間 內文 與 色彩 
                text_surface = my_font.render(f'Points: {format(points)}', True, (0, 255, 0))
                text_time = my_font.render(f'time: {format(round(T))}', True, (0, 255, 0))

                # 正式放入 雷電
                window_surface.blit(light.image, light.rect)

                # 正式放入 文字
                window_surface.blit(text_surface, (10, 0))
                window_surface.blit(text_time, (10, 40))

                # 循環更新
                pygame.display.update()
            

    
##############################################################################

# 評分部分
##############################################################################
            while Score:

                    window_surface.fill([255, 255, 255])
                    window_surface.blit(bg_score, (0, 0))

                    button_menu.draw(window_surface)
                    button_again.draw(window_surface)
                    button_exit.draw(window_surface)

                    CS = list(cursor.execute('SELECT * FROM `ninja_war` WHERE `name` = ? Order BY `time` DESC',[User]))
                    if len(CS):
                        current_score = my_font.render(f'目前分數:   {CS[0][2]}分   {TW_Time(CS[0][3])[:10]}', True, (0, 255, 0))
                        window_surface.blit(current_score, (200, 70))

                    HS = list(cursor.execute('SELECT * FROM `ninja_war` WHERE `name` = ? Order BY `score` DESC',[User]))
                    if len(CS):
                        highest_score = my_font.render(f'最高分數:   {HS[0][2]}分   {TW_Time(HS[0][3])[:10]}', True, (255, 127, 80))
                        window_surface.blit(highest_score, (200, 150))

                    if len(CS) >= 4:
                        History = my_font.render('歷史分數', True, (30, 144, 255))
                        window_surface.blit(History, (200, 230))
                        for i in range(1,4):
                            History_score = my_font.render(f'前 {i} 次分數:   {CS[i][2]}分   {TW_Time(CS[i][3])[:10]}', True, (30, 144, 255))
                            window_surface.blit(History_score, (200, 230+80*i))
                    elif len(CS) == 3:
                        History = my_font.render('歷史分數', True, (30, 144, 255))
                        window_surface.blit(History, (200, 230))
                        for i in range(1,3):
                            History_score = my_font.render(f'前 {i} 次分數:   {CS[i][2]}分   {TW_Time(CS[i][3])[:10]}', True, (30, 144, 255))
                            window_surface.blit(History_score, (200, 230+80*i))
                    elif len(CS) == 2:
                        History = my_font.render('歷史分數', True, (30, 144, 255))
                        window_surface.blit(History, (200, 230))
                        for i in range(1,2):
                            History_score = my_font.render(f'前 {i} 次分數:   {CS[i][2]}分   {TW_Time(CS[i][3])[:10]}', True, (30, 144, 255))
                            window_surface.blit(History_score, (200, 230+80*i))

                    mouse_pos = pygame.mouse.get_pos()       

                    for event in pygame.event.get():

                        # 關閉事件1 (按下右上叉叉，離開遊戲)
                        if event.type == pygame.QUIT:
                            On_status = False
                            sys.exit(0)
                            camera.release()




                        # 鍵盤事件
                        elif event.type == KEYDOWN:
                            if event.key == K_ESCAPE or event.key == K_v:
                                On_status = False
                                sys.exit(0)
                                camera.release()

                            elif event.key == K_f:
                                Fullscreen = not Fullscreen
                                if Fullscreen:
                                    screen = pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                                else:
                                    screen = pygame.display.set_mode((1280, 720), 0, 32)        

                            elif event.key == K_z:
                                Menu = True
                                Run = False
                                Score = False
                                Demo = False

                            elif event.key == K_x:
                                Menu = False
                                Run = True
                                Score = False
                                Demo = False

                            elif event.key == K_f:
                                Fullscreen = not Fullscreen
                                if Fullscreen:
                                    screen = pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                                else:
                                    screen = pygame.display.set_mode((1280, 720), 0, 32)

                        elif event.type == MOUSEBUTTONDOWN:
                            if button_menu.rect.collidepoint(mouse_pos):
                                Menu = True
                                Run = False
                                Score = False
                                Demo = False

                            elif button_again.rect.collidepoint(mouse_pos):
                                Menu = False
                                Run = True
                                Score = False
                                Demo = False

                            elif button_exit.rect.collidepoint(mouse_pos):
                                On_status = False
                                sys.exit(0)
                                camera.release()


                    pygame.display.update()
                
                
                

                
                
                
                
# 進入demo
#############################################################
            demo_video = cv2.VideoCapture('NINJA_WAR/button/DEMO_NINJA_WARS.mp4')
            while Demo:
                
                hasFrame, img = demo_video.read()
                time.sleep(0.001)    
                if not hasFrame:
                    Menu = True
                    Run = False
                    Score = False
                    Demo = False
                    break
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.swapaxes(0, 1)
        
        
                pygame.surfarray.blit_array(window_surface, img)
        
                button_menu.draw(window_surface)

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

                        elif event.key == K_f:
                            Fullscreen = not Fullscreen
                            if Fullscreen:
                                screen = pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                            else:
                                screen = pygame.display.set_mode((1280, 720), 0, 32)

                    elif event.type == MOUSEBUTTONDOWN:
                        if button_menu.rect.collidepoint(mouse_pos):
                            Menu = True
                            Run = False
                            Score = False
                            Demo = False
                            


                            
#                     elif event.type == select_button:
#                         if pygame.sprite.collide_rect(button_menu, bonber):
#                             Menu = True
#                             Run = False
#                             Score = False
#                             Demo = False
                            

                pygame.display.update()


    except (KeyboardInterrupt, SystemExit):
        pygame.quit()
        cv2.destroyAllWindows()
        camera.release()                          
                        
if __name__ == '__main__':    
    main()
    






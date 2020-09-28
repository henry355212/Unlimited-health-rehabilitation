from pygame.locals import MOUSEBUTTONDOWN ,KEYDOWN, K_ESCAPE, K_q, K_x, K_f, K_z, K_c, K_v, FULLSCREEN
import pygame
from pygame.locals import Color, QUIT, MOUSEBUTTONDOWN, USEREVENT, USEREVENT
from pygame.sprite import collide_rect, Sprite, spritecollide
import cv2
import sys
import tensorflow as tf
import numpy as np
import time
import random
import imutils
import pickle
import os
from subprocess import run
import sqlite3
from pygame import mixer
import tempfile
from gtts import gTTS

db = sqlite3.connect('game.db')
cursor = db.cursor()



# 視窗大小
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# 按鈕 (x,y)在中心
############################################################
class Button(pygame.sprite.Sprite):
    def __init__(self, x, y, picture,ratio):
        super().__init__()
        pygame.sprite.Sprite.__init__(self)
        self.ratio = ratio
        self.x = x
        self.y = y
        self.raw_image = pygame.image.load(picture)
        self.width = cv2.imread(picture).shape[1]
        self.height = cv2.imread(picture).shape[0]
        self.image = pygame.transform.scale(self.raw_image, (int(self.width*self.ratio), int(self.height*self.ratio)))
        self.rect = self.image.get_rect()
        self.rect.center = (self.x, self.y)
        
    def draw(self, win):
        win.blit(self.image, self.rect)
############################################################

# 圖片 (x,y)在右上
############################################################
class Fig(pygame.sprite.Sprite):
    def __init__(self, x, y, picture,ratio):
        super().__init__()
        pygame.sprite.Sprite.__init__(self)
        self.ratio = ratio
        self.x = x
        self.y = y
        self.raw_image = pygame.image.load(picture)
        self.width = cv2.imread(picture).shape[1]
        self.height = cv2.imread(picture).shape[0]
        self.image = pygame.transform.scale(self.raw_image, (int(self.width*self.ratio), int(self.height*self.ratio)))
        self.rect = self.image.get_rect()
        self.rect.topright = (self.x, self.y)
        
    def draw(self, win):
        win.blit(self.image, self.rect)
############################################################


# 人臉辨識相關模型導入
#############################################################
# 偵測人臉部分
protoPath = 'face_detection_model/deploy.prototxt'
modelPath = 'face_detection_model/res10_300x300_ssd_iter_140000_fp16.caffemodel'
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# 特徵擷取部分
embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

# 預訓練模型與標記
recognizer = pickle.loads(open('output/recognizer.pickle', "rb").read())
le = pickle.loads(open('output/le.pickle', "rb").read())

#  人臉辨識 wabcam 
FR = cv2.VideoCapture(0)

# vs = VideoStream(src=0).start()
############################################################


def login_interface(login, face_reg, main_menu, window_surface):
    while login:

        mouse_pos = pygame.mouse.get_pos()

        window_surface.blit(pygame.image.load('img_face_login/bg_login.jpg'), (0, 0))

        # 設定按鈕
        b_face_login = Button(640, 300, 'img_face_login/face_login.png',0.66)
        b_other_login = Button(640, 450, 'img_face_login/other_login.png',0.66)
        b_exit = Button(1150, 650, 'img_face_login/exit.png',0.66)

        b_face_login.draw(window_surface)
        b_other_login.draw(window_surface)
        b_exit.draw(window_surface)


        #事件迴圈
        for event in pygame.event.get():

            # 關閉事件 (按下右上叉叉，離開遊戲)
            if event.type == pygame.QUIT:
                login = False
                sys.exit(0)


            # 鍵盤事件
            elif event.type == KEYDOWN:

                # 按下ESC or v，離開遊戲
                if event.key == K_ESCAPE or event.key == K_v:
                    login = False
                    sys.exit(0)

                # 按下F，切換全螢幕
                elif event.key == K_f:
                    Fullscreen = not Fullscreen
                    if Fullscreen:
                        screen = pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                    else:
                        screen = pygame.display.set_mode((1280, 720), 0, 32)

            elif event.type == MOUSEBUTTONDOWN:

                # 按下start按鈕，開始遊戲
                if b_face_login.rect.collidepoint(mouse_pos):
                    login = False
                    face_reg = True
                    main_menu = False


                        # 按下score按鈕，積分頁面
                elif b_other_login.rect.collidepoint(mouse_pos):

                    main_menu = True
                    face_reg = False
                    login = False
                    User = 'guest'




                        # 按下demo按鈕，開始Demo    
                elif b_exit.rect.collidepoint(mouse_pos):
                    login = False
                    sys.exit(0)

            pygame.display.update()





def main():           
    try:
        # 初始化
        pygame.init()
        
        # 視窗畫面大小
        window_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # 遊戲名稱
        pygame.display.set_caption('game')

        # 遊戲icon
        logo = pygame.image.load('game.png')
        pygame.display.set_icon(logo)
        
        # 起始畫面白色
        window_surface.fill([255, 255, 255])
        
        
        # 導入全螢幕設定
        Fullscreen = False
        
        # 文字格式
        ############################################################
        my_font = pygame.font.Font('chinese.msyh.ttf', 60)
        ############################################################
        
        
        login = True


        while login:

            mouse_pos = pygame.mouse.get_pos()

            window_surface.blit(pygame.image.load('img_face_login/bg_login.jpg'), (0, 0))

            # 設定按鈕
            b_face_login = Button(640, 300, 'img_face_login/face_login.png',0.66)
            b_other_login = Button(640, 450, 'img_face_login/other_login.png',0.66)
            b_exit = Button(1150, 650, 'img_face_login/exit.png',0.66)
            

            b_face_login.draw(window_surface)
            b_other_login.draw(window_surface)
            b_exit.draw(window_surface)


            #事件迴圈
            for event in pygame.event.get():

                # 關閉事件 (按下右上叉叉，離開遊戲)
                if event.type == pygame.QUIT:
                    login = False
                    sys.exit(0)


                # 鍵盤事件
                elif event.type == KEYDOWN:

                    # 按下ESC or v，離開遊戲
                    if event.key == K_ESCAPE or event.key == K_v:
                            login = False
                            sys.exit(0)

                    # 按下F，切換全螢幕
                    elif event.key == K_f:
                        Fullscreen = not Fullscreen
                        if Fullscreen:
                            screen = pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                        else:
                            screen = pygame.display.set_mode((1280, 720), 0, 32)

                elif event.type == MOUSEBUTTONDOWN:

                    # 按下start按鈕，開始遊戲
                    if b_face_login.rect.collidepoint(mouse_pos):
                        login = False
                        face_reg = True
                        main_menu = False


                        # 按下score按鈕，積分頁面
                    elif b_other_login.rect.collidepoint(mouse_pos):
                        User = 'guest'
                        ch_name = 'guest'
                        user_image = Fig(1280, 0,'face_img/guest.png',1)
                        f = open('User.txt','w+')
                        f.write(User)
                        f.close()
                        main_menu = True
                        face_reg = False
                        login = False





                        # 按下demo按鈕，開始Demo    
                    elif b_exit.rect.collidepoint(mouse_pos):
                        login = False
                        sys.exit(0)

                pygame.display.update()
        
        
        while face_reg:
        
            
            
            # 讀取鏡頭與前處裡
#             frame = vs.read() 
            ret, frame = FR.read()
            frame = imutils.resize(frame, width=1280)
            (h, w) = frame.shape[:2]
            imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                              1.0, (300, 300),(104.0, 117.0, 123.0), 
                                              swapRB=False, crop=False)
            
            # 人臉偵測
            detector.setInput(imageBlob)
            detections = detector.forward()
            
            
            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI
                    face = frame[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                        continue

                    # construct a blob for the face ROI, then pass the blob
                    # through our face embedding model to obtain the 128-d
                    # quantification of the face
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                        (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()

                    # perform classification to recognize the face
                    preds = recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = le.classes_[j]
                    text = "{}: {:.2f}%".format(name, proba * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.swapaxes(0, 1)
                    frame = frame[:, 120:840]
                    pygame.surfarray.blit_array(window_surface, frame)
                    if proba>0.7:
                        user_image = Fig(1280, 0, f'face_img/{name}.png',0.2)
                        User = name
                        
                        f = open('User.txt','w+')
                        f.write(User)
                        f.close()
                        
                        ch_name = list(cursor.execute("SELECT `ch_name` FROM `user_table` WHERE `en_name` = (?)",[User]))[0][0]
                        FR.release()
                        main_menu = True
                        face_reg = False
                        
                    

                
                        
                        
            
            
            #事件迴圈
            for event in pygame.event.get():
                
                
                
                # 關閉事件 (按下右上叉叉，離開遊戲)
                if event.type == pygame.QUIT:
                    FR.release()
                    face_reg = False
                    sys.exit(0)
                    
                # 鍵盤事件
                elif event.type == KEYDOWN:
                        
                    # 按下ESC or v，離開遊戲
                    if event.key == K_ESCAPE or event.key == K_v:
                        FR.release()
                        face_reg = False
                        sys.exit(0)
                        
                    # 按下F，切換全螢幕
                    elif event.key == K_f:
                        Fullscreen = not Fullscreen
                        if Fullscreen:
                            screen = pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                        else:
                            screen = pygame.display.set_mode((1280, 720), 0, 32)
            
            pygame.display.update()
            
        tts=gTTS(text=(f'Hi {ch_name}'), lang='zh-tw')
        filename = tempfile.NamedTemporaryFile().name+'.mp3'
        tts.save(filename)
            
        mixer.init()
        mixer.music.load(filename)
        mixer.music.play(0)
            
        while main_menu:
        
            window_surface.blit(pygame.image.load('img_face_login/bg_main.jpg'), (0, 0))
            user_image.draw(window_surface)
            
            tag_user = my_font.render(f'Hello {ch_name}', True, (255, 255, 255))
            window_surface.blit(tag_user, (700, 0))


            #tts=gTTS(text=ch_name, lang='zh-tw')
            #filename = tempfile.NamedTemporaryFile().name+'.mp3'
            #tts.save(filename)


            #music = True
            #while music:
                #播放音ＴＳＯ
                # mixer.init()
                # mixer.music.load(filename)
                #mixer.music.play(0)
                #music = False
            
            b_UFO_WAR = Button(250, 100, 'img_face_login/UFO_WARS.png',0.6)
            b_UFO_WAR.draw(window_surface)
            
            b_HAPPY_BIRD = Button(250, 220, 'img_face_login/HAPPY_BIRD.png',0.6)
            b_HAPPY_BIRD.draw(window_surface)
            
            b_LASER_EYES = Button(250, 340, 'img_face_login/LASER_EYES.png',0.6)
            b_LASER_EYES.draw(window_surface)
            
            b_NINJA_WARS = Button(250, 460, 'img_face_login/NINJA_WARS.png',0.6)
            b_NINJA_WARS.draw(window_surface)
            
            b_NORA = Button(250, 580, 'img_face_login/MORA.png',0.6)
            b_NORA.draw(window_surface)
            
            
            #事件迴圈
            for event in pygame.event.get():

                mouse_pos = pygame.mouse.get_pos()

                # 關閉事件 (按下右上叉叉，離開遊戲)
                if event.type == pygame.QUIT:
                    main_menu = False
                    sys.exit(0)
                    
                # 鍵盤事件
                elif event.type == KEYDOWN:
                        
                    # 按下ESC or v，離開遊戲
                    if event.key == K_ESCAPE or event.key == K_v:
                        main_menu = False
                        sys.exit(0)
                        
                    # 按下F，切換全螢幕
                    elif event.key == K_f:
                        Fullscreen = not Fullscreen
                        if Fullscreen:
                            screen = pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                        else:
                            screen = pygame.display.set_mode((1280, 720), 0, 32)

                elif event.type == MOUSEBUTTONDOWN:

                    if b_UFO_WAR.rect.collidepoint(mouse_pos):
                        run('python UFO_WAR.py').stdout
                        
                    elif b_HAPPY_BIRD.rect.collidepoint(mouse_pos):
                        run('python happy_bird_V4_end.py').stdout

                    elif b_LASER_EYES.rect.collidepoint(mouse_pos):
                        run('python lasereye.py').stdout

                    elif b_NINJA_WARS.rect.collidepoint(mouse_pos):
                        run('python NinjaWar.py').stdout

                    elif b_NORA.rect.collidepoint(mouse_pos):
                        run('python MORA.py').stdout


            
            pygame.display.update()
        
    except (KeyboardInterrupt, SystemExit):
        pygame.quit()
        cv2.destroyAllWindows()
        
        
if __name__ == '__main__':    
    main()

import sys
import math
import random


import numpy as np
from numpy import linalg as LA
import cv2
from scipy.spatial import distance
from openvino.inference_engine import IECore
from openvino.inference_engine import IENetwork


from pygame.locals import KEYDOWN, K_ESCAPE, K_q, K_x, K_f, K_z, K_c, K_v, FULLSCREEN
import pygame
from pygame.locals import Color, QUIT, MOUSEBUTTONDOWN, USEREVENT, USEREVENT
from pygame.sprite import collide_rect, Sprite, spritecollide
import cv2
import time
import os
import sqlite3
from datetime import datetime,timezone,timedelta

# 連結Database
db = sqlite3.connect('game.db')
cursor = db.cursor()

# 讀取User資訊
with open('User.txt','r') as f:
    test = f.read().splitlines()
User = test[0]

#模型設定
###############################################################################################
model_det  = 'face-detection-adas-0001'
model_hp   = 'head-pose-estimation-adas-0001'
model_gaze = 'gaze-estimation-adas-0002'
model_lm   = 'facial-landmarks-35-adas-0002'

model_det  = './intel/'+model_det +'/FP16/'+model_det
model_hp   = './intel/'+model_hp  +'/FP16/'+model_hp
model_gaze = './intel/'+model_gaze+'/FP16/'+model_gaze
model_lm   = './intel/'+model_lm  +'/FP16/'+model_lm


#遊戲視窗大小
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# 目標大小
IMAGEWIDTH = 150
IMAGEHEIGHT = 100

# 按鈕大小
BUTTONWIDTH = 256
BUTTONHEIGHT = 128

# 爆炸
EXPLOSIONWIDTH = 300
EXPLOSIONHEIGHT = 200


# N代表數量， C代表channel，H代表高度，W代表寬度.
_N = 0
_C = 1
_H = 2
_W = 3

	
#畫線	
###############################################################################################
def draw_gaze_line(img, coord1, coord2, laser_flag):
    if laser_flag == True:
        # simple line
        cv2.line(img, coord1, coord2, (0, 0, 255),2)

        

# 目標部分
##################################################################################################
class Target(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        pygame.sprite.Sprite.__init__(self)
        self.x = random.randrange(0, 1280-IMAGEWIDTH)
        self.y = random.randrange(0, 720-IMAGEHEIGHT)
        self.raw_image = pygame.image.load('Laser_EYE/SpaceInvaders.png')
        self.image = pygame.transform.scale(self.raw_image, (IMAGEWIDTH, IMAGEHEIGHT))
        self.rect = self.image.get_rect()
        self.rect.topleft = (self.x, self.y)
        self.width = IMAGEWIDTH
        self.height = IMAGEHEIGHT
        
    def draw(self, win):
        win.blit(self.image, self.rect)
    def move(self):
        self.x += random.randrange(-10, 10)
        self.y += random.randrange(-10, 10)
        self.rect.topleft = (self.x, self.y)



# 爆炸部分
################################################################################################
class Explosion(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.raw_image = pygame.image.load("Laser_EYE/explosion.png")
        self.image = pygame.transform.scale(self.raw_image, (EXPLOSIONWIDTH, EXPLOSIONHEIGHT))
        self.rect = self.image.get_rect()
        self.rect.center = (self.x, self.y)
        self.width = EXPLOSIONWIDTH
        self.height = EXPLOSIONHEIGHT
        self.widow_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT

    def draw(self, win):
        win.blit(self.image, self.rect)



# 時區轉換
def TW_Time(T):
    dt1 = datetime.strptime(T, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    return dt2.strftime("%Y-%m-%d %H:%M:%S")



#取起點跟終點間的每一個點
################################################################################################
def point_line(x1,y1,x2,y2):
    px=[]
    if abs(x2-x1)>=abs(y2-y1):
        if x2+1 >= x1:
            for i in range(x1,x2+1):
                px.append((i,(y1+round((i-x1)*(y2-y1)/(x2-x1)))))
        else :
            for i in range(x2,x1+1):
                px.append((i,(y1+round((i-x1)*(y2-y1)/(x2-x1)))))


    else:
        if y2+1>=y1:
            for i in range(y1,y2+1):
                px.append(((x1+round((i-y1)*(x2-x1)/(y2-y1))),i))
        else:
            for i in range(y2,y1+1):
                px.append(((x1+round((i-y1)*(x2-x1)/(y2-y1))),i))

    return px


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

        
#按鈕圖示
################################################################################################
button_start = Button(640, 360, 'Laser_EYE/start.png')
button_exit = Button(1100, 650, 'Laser_EYE/exit.png')
button_demo = Button(290, 360, 'Laser_EYE/demo.png')
button_score = Button(990, 360, 'Laser_EYE/score.png')
button_menu = Button(180, 650, 'Laser_EYE/menu.png')
button_again = Button(640, 650, 'Laser_EYE/again.png')


# 起始背景
bg = pygame.image.load('Laser_EYE/space1920.jpg')
bg_score = pygame.image.load('Laser_EYE/loSpace1920.png')


#在黑窗顯示字
def usage():
    print("""
Gaze estimation demo
'q': Exit
""")



def main():

        try:
            usage()
    
            boundary_box_flag = False
    
            #模型匯入
            ###############################################################################################
            
            ie = IECore() #model 匯入
            
            # 人臉偵測
            net_det  = IENetwork(model=model_det+'.xml', weights=model_det+'.bin')  # 載入模型及權重:
            input_name_det  = next(iter(net_det.inputs))                            # Input blob name "data" 設置輸入空間
            input_shape_det = net_det.inputs[input_name_det].shape                  # [1,3,384,672] 取得批次數量、通道數及影像高、寬
            out_name_det    = next(iter(net_det.outputs))                           # Output blob name "detection_out" 設置輸出空間
            exec_net_det    = ie.load_network(network=net_det, device_name='CPU', num_requests=1) #載入模型到指定裝置
            del net_det
    
            # 人臉座標
            net_lm = IENetwork(model=model_lm+'.xml', weights=model_lm+'.bin')      # 人臉座標
            input_name_lm  = next(iter(net_lm.inputs))                              # Input blob name 
            input_shape_lm = net_lm.inputs[input_name_lm].shape                     # [1,3,60,60]
            out_name_lm    = next(iter(net_lm.outputs))                             # Output blob name "embd/dim_red/conv"
            out_shape_lm   = net_lm.outputs[out_name_lm].shape                      # 3x [1,1]
            exec_net_lm    = ie.load_network(network=net_lm, device_name='CPU', num_requests=1)
            del net_lm
    
            # 頭部姿勢
            net_hp = IENetwork(model=model_hp+'.xml', weights=model_hp+'.bin')      # 頭部姿勢
            input_name_hp  = next(iter(net_hp.inputs))                              # Input blob name
            input_shape_hp = net_hp.inputs[input_name_hp].shape                     # [1,3,60,60]
            out_name_hp    = next(iter(net_hp.outputs))                             # Output blob name
            out_shape_hp   = net_hp.outputs[out_name_hp].shape                      # [1,70]
            exec_net_hp    = ie.load_network(network=net_hp, device_name='CPU', num_requests=1)
            del net_hp
    
            #視線偵測
            net_gaze = IENetwork(model=model_gaze+'.xml', weights=model_gaze+'.bin')#視線偵測
            input_shape_gaze  = [1, 3, 60, 60]
            exec_net_gaze     = ie.load_network(network=net_gaze, device_name='CPU')
            del net_gaze
    
    
            
            # 相機設定
        ##############################################################################################################################################
            cam = cv2.VideoCapture(0)
            camx, camy = (WINDOW_WIDTH, WINDOW_HEIGHT)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH , camx)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camy)
            
    
    
            #遊戲設定
            ##############################################################################################################################################		
            # 初始化
            pygame.init()
            
            # 視窗畫面大小
            window_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            
            # 遊戲名稱
            pygame.display.set_caption('Laser Eye')

            # 遊戲icon
            logo = pygame.image.load('Laser_EYE/SpaceInvaders-icon.ico')
            pygame.display.set_icon(logo)

            # 全螢幕
            Fullscreen = False
    
            # 目標參數設定
            target = Target()
    
            # 分數起始值
            points = 0
            
            # 時間起始值
            T = 30
            
            # 各文字字形與大小設定
            my_font = pygame.font.Font('chinese.msyh.ttf', 60)
            my_hit_font = pygame.font.Font('chinese.msyh.ttf', 240)
    
    
    
    
            # 事件設定
            ################################################################################# 
            # 1. 移動 
            reload_ufo_event = USEREVENT+1
            pygame.time.set_timer(reload_ufo_event, 300) # 設定每300毫秒更新一次
            
            # 2. 碰撞
            kill_event = USEREVENT+2
            pygame.time.set_timer(kill_event, 100) # 設定每100毫秒更新一次
    
            # 3. gameover
            gameover = USEREVENT+3
            pygame.time.set_timer(gameover, 100) # 設定每100毫秒更新一次 
    
            
            # 進入Pygame      
            #################################################################################
            Menu = True
            laser_flag = True #視線
            flip_flag = True #翻轉
            On_status = True
            while On_status:
                
                # 進入選單 
                #################################################################################   
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
                                
                            # 按下ESC or v，離開遊戲
                            if event.key == K_ESCAPE or event.key == K_v:
                                On_status = False
                                sys.exit(0)
                                camera.release()
                                
                            # 按下x，開始遊戲
                            elif event.key == K_x:
                                Menu = False
                                Run = True
                                Score = False
                                Demo = False
                                    
                            # 按下z，開始Demo    
                            elif event.key == K_z:
                                Menu = False
                                Run = False
                                Score = False
                                Demo = True
                                
                            # 按下c，積分頁面
                            elif event.key == K_c:
                                Menu = False
                                Run = False
                                Score = True
                                Demo = False
                                
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
    ########    ###################################################################################################           
        
        
        
                #進入遊戲
                ###################################################################################################			       
        
                while Run:
                    ret,img = cam.read()#相機讀取
                    if ret==False:      #未偵測到相機就跳出
                        break
        
                    if flip_flag == True:
                        img = cv2.flip(img, 1)  #翻轉畫面
                    out_img = img.copy()        #複製一份原圖檔，將原圖丟進去運算，用複製的圖來作畫
        
                    #開始對原圖做運算
                ####################################################################################################		
                    img1 = cv2.resize(img, (input_shape_det[_W], input_shape_det[_H]))
                    img1 = img1.transpose((2, 0, 1))                                       # Change data layout from HWC to CHW 
                    img1 = img1.reshape(input_shape_det)
                    res_det = exec_net_det.infer(inputs={input_name_det: img1})            # Detect faces	
                    global Lefteye,Leftlaser,Righteye,Rightlaser
        
                    Lline = []
                    Rline = []
                    for obj in res_det[out_name_det][0][0]:                                # obj = [ image_id, label, conf, xmin, ymin, xmax, ymax ]
                        if obj[2] > 0.75:                                                  # Confidence > 75% 可信度
                            xmin = abs(int(obj[3] * img.shape[1]))
                            ymin = abs(int(obj[4] * img.shape[0]))
                            xmax = abs(int(obj[5] * img.shape[1]))
                            ymax = abs(int(obj[6] * img.shape[0]))
                            class_id = int(obj[1])
                            face=img[ymin:ymax,xmin:xmax]                                  # Crop the face image
                            if boundary_box_flag == True:                                  #畫臉框
                                cv2.rectangle(out_img, (xmin, ymin), (xmax, ymax), (255,255,0), 2)
        
        
                            #設人臉座標
                            ######################################################################################################
                            face1=cv2.resize(face, (input_shape_lm[_W], input_shape_lm[_H]))
                            face1=face1.transpose((2,0,1))
                            face1=face1.reshape(input_shape_lm)
                            res_lm = exec_net_lm.infer(inputs={input_name_lm: face1})       # Run landmark detection
                            lm=res_lm[out_name_lm][0][:8].reshape(4,2)                      #  [[left0x, left0y], [left1x, left1y], [right0x, right0y], [right1x, right1y] ]
        
                            # 偵測頭部的姿勢 (yaw=Y軸, pitch=X軸, role=Z軸)的旋轉
                            res_hp = exec_net_hp.infer(inputs={input_name_hp: face1})
                            yaw   = res_hp['angle_y_fc'][0][0]
                            pitch = res_hp['angle_p_fc'][0][0]
                            roll  = res_hp['angle_r_fc'][0][0]
        
                            _X=0
                            _Y=1
                            
                            # Landmark position memo...   lm[1] (eye) lm[0] (nose)  lm[2] (eye) lm[3]
                            
                            # 裁切臉部圖像中的眼睛大小
                            eye_sizes   = [ abs(int((lm[0][_X]-lm[1][_X]) * face.shape[1])), abs(int((lm[3][_X]-lm[2][_X]) * face.shape[1])) ]
                            # 抓臉部圖像中的眼睛的中心點
                            eye_centers = [ [ int(((lm[0][_X]+lm[1][_X])/2 * face.shape[1])), int(((lm[0][_Y]+lm[1][_Y])/2 * face.shape[0])) ], 
                                            [ int(((lm[3][_X]+lm[2][_X])/2 * face.shape[1])), int(((lm[3][_Y]+lm[2][_Y])/2 * face.shape[0])) ] ]  
                            if eye_sizes[0]<4 or eye_sizes[1]<4:
                                continue
        
                            ratio = 0.7
                            eyes = []
                            
                            for i in range(2):
                                # Crop eye images
                                x1 = int(eye_centers[i][_X]-eye_sizes[i]*ratio)
                                x2 = int(eye_centers[i][_X]+eye_sizes[i]*ratio)
                                y1 = int(eye_centers[i][_Y]-eye_sizes[i]*ratio)
                                y2 = int(eye_centers[i][_Y]+eye_sizes[i]*ratio)
                                eyes.append(cv2.resize(face[y1:y2, x1:x2].copy(), (input_shape_gaze[_W], input_shape_gaze[_H])))    # crop and resize
        
                                #畫眼框 boundary boxes
                                if boundary_box_flag == True:
                                    cv2.rectangle(out_img, (x1+xmin,y1+ymin), (x2+xmin,y2+ymin), (0,255,0), 2)
        
                            #rotate eyes around Z axis to keep them level
                                if roll != 0.:
                                    rotMat = cv2.getRotationMatrix2D((int(input_shape_gaze[_W]/2), int(input_shape_gaze[_H]/2)), roll, 1.0)
                                    eyes[i] = cv2.warpAffine(eyes[i], rotMat, (input_shape_gaze[_W], input_shape_gaze[_H]), flags=cv2.INTER_LINEAR)
                                eyes[i] = eyes[i].transpose((2, 0, 1))                                     # Change data layout from HWC to CHW
                                eyes[i] = eyes[i].reshape((1,3,60,60))
        
                            hp_angle = [ yaw, pitch, 0 ]                                                   # head pose angle in degree
                            res_gaze = exec_net_gaze.infer(inputs={'left_eye_image'  : eyes[0], 
                                                                'right_eye_image' : eyes[1],
                                                                'head_pose_angles': hp_angle})          # gaze estimation
                            gaze_vec = res_gaze['gaze_vector'][0]                                          # result is in orthogonal coordinate system (x,y,z. not yaw,pitch,roll)and not normalized
                            gaze_vec_norm = gaze_vec / np.linalg.norm(gaze_vec)                            # normalize the gaze vector
        
                            vcos = math.cos(math.radians(roll))
                            vsin = math.sin(math.radians(roll))
                            tmpx =  gaze_vec_norm[0]*vcos + gaze_vec_norm[1]*vsin
                            tmpy = -gaze_vec_norm[0]*vsin + gaze_vec_norm[1]*vcos
                            gaze_vec_norm = [tmpx, tmpy]
        
        
        
                            # 抓視線的起點位置跟終點位置並畫線
                            
                            # 左眼起點(x,y)終點(x,y)
                            Lefteye = (eye_centers[0][_X]+xmin,                                 eye_centers[0][_Y]+ymin) 
                            Leftlaser = (eye_centers[0][_X]+xmin+int((gaze_vec_norm[0]+0.)*1000), eye_centers[0][_Y]+ymin-int((gaze_vec_norm[1]+0.)*1000))
        
                            # 右眼起點(x,y)終點(x,y)
                            Righteye = (eye_centers[1][_X]+xmin,                                 eye_centers[1][_Y]+ymin)
                            Rightlaser = (eye_centers[1][_X]+xmin+int((gaze_vec_norm[0]+0.)*1000), eye_centers[1][_Y]+ymin-int((gaze_vec_norm[1]+0.)*1000))
        
                            #print("Lefteye:",Lefteye,Leftlaser)
                            #print("Leftlaser:",Leftlaser)
                            #print("Refteye:",Righteye,Rightlaser)
                            #print("Reftlaser:",Reftlaser)
                            
                            Lline = point_line(Lefteye[0],Lefteye[1],Leftlaser[0],Leftlaser[1])         # 左眼起點到終點的每一點(x,y)
                            
                            Rline = point_line(Righteye[0],Righteye[1],Rightlaser[0],Rightlaser[1])     # 右眼起點到終點的每一點(x,y)
                            
                            #print("Lline:",Lline)
                            #print("==========================================================")
                            #print("Rline:",Rline)
        
        
        
                    # 正式放入webcam影像
                    window_surface.fill([255, 255, 255])
                    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
                    out_img = out_img.swapaxes(0, 1)  #畫面顛倒
                    
                    pygame.surfarray.blit_array(window_surface, out_img)
        
        
                    # 爆炸設定
                    explosion = Explosion(target.x+IMAGEWIDTH/2,target.y+IMAGEHEIGHT/2)
        
        
                    #將目標放入畫布
                    target.draw(window_surface)
                    pygame.draw.line(window_surface, (0,255,0), Lefteye, Leftlaser, 7)
                    pygame.draw.line(window_surface, (0,255,0), Righteye, Rightlaser, 7)
        
                    
                    
                    
                    T -= 0.1 # 時間累計
        
                                
                    # 遊戲分數與時間 內文 與 色彩 
                    text_surface = my_font.render(f'Points: {format(points)}', True, (0, 255, 0))
                    text_time = my_font.render(f'time: {format(round(T))}', True, (0, 255, 0))
                    
                    
                    # 正式放入 文字
                    window_surface.blit(text_surface, (10, 0))
                    window_surface.blit(text_time, (10, 40))
                    
                    
                    
                    # 正式處理事件
                    for event in pygame.event.get():
        
                        # 目標移動 
                        if event.type == reload_ufo_event:
                            target.move()
                            
                        # 目標消滅
                        elif event.type == kill_event:
                            for i in Lline:
                                #print("Lline:",i)
                                if target.rect.collidepoint(i):
                                    explosion.draw(window_surface)
                                    target.kill()
                                    target = Target()
                                    points += 5  # 分數累計
                                    
                            for i in Rline:
                                #print("Rline:",i)
                                if target.rect.collidepoint(i):
                                    explosion.draw(window_surface)
                                    target.kill()
                                    target = Target()
                                    points += 5 # 分數累計
                                    
        
                        #計時器
                        elif event.type == gameover:
                            if T <= 0:
                                print(T)
                                print(points)
                                print(User)
                                cursor.execute('INSERT INTO `laser_eye`( `name`, `score`) VALUES (?,?)',[User,points])
                                db.commit()
                                target.kill()
                                target = Target()
                                points = 0
                                T = 30
                                Menu = False
                                Run = False
                                Score = True
                                Demo = False
        
                        # 關閉事件1 (按下右上叉叉，離開遊戲)
                        elif event.type == pygame.QUIT:
                            On_status = False
                            sys.exit(0)
                            camera.release()
                            
                        # 關閉事件2 (按下ESC 或 Q，離開遊戲)
                        elif event.type == KEYDOWN:
                            if event.key == K_ESCAPE or event.key == K_q:
                                On_status = False
                                sys.exit(0)
                                camera.release()
                            elif event.key == K_z:
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
        
                    
                    # 循環更新
                    pygame.display.update()
            
            
            
        
                # 進入評分
                #####################################################################################
                while Score:
                
                    #白底畫布
                    window_surface.fill([255, 255, 255])
                    window_surface.blit(bg_score, (0, 0))
    
                    #按鈕
                    button_menu.draw(window_surface)
                    button_again.draw(window_surface)
                    button_exit.draw(window_surface)
                
                    
                    #資料寫入DB中
                    CS = list(cursor.execute('SELECT * FROM `laser_eye` WHERE `name` = ? Order BY `time` DESC',[User]))
                    if len(CS):
                        current_score = my_font.render(f'目前分數:   {CS[0][2]}分   {TW_Time(CS[0][3])[:10]}', True, (0, 255, 0))
                        window_surface.blit(current_score, (200, 70))
            
                    HS = list(cursor.execute('SELECT * FROM `laser_eye` WHERE `name` = ? Order BY `score` DESC',[User]))
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
        
                        # 關閉事件2 (按下ESC 或 Q，離開遊戲)
                        elif event.type == KEYDOWN:
                            if event.key == K_ESCAPE or event.key == K_v:
                                On_status = False
                                sys.exit(0)
                                camera.release()
        
                            elif event.key == K_z: #回到MENU
                                Menu = True
                                Run = False
                                Score = False
                                Demo = False
                                
                            elif event.key == K_x: #再一次
                                Menu = False
                                Run = True
                                Score = False
                                Demo = False
        
                            elif event.key == K_f: #全螢幕
                                Fullscreen = not Fullscreen
                                if Fullscreen:
                                    screen = pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                                else:
                                    screen = pygame.display.set_mode((1280, 720), 0, 32)
        
                        #偵測滑鼠
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
    
                
                demo_video=cv2.VideoCapture('Laser_EYE/DEMO_LASER_EYES.mp4')
                while Demo :
                    hasFrame, img=demo_video.read()
                    time.sleep(0.001)
                    if not hasFrame:
                        Menu=True
                        Run=False
                        Score=False
                        Demo=False
                        break
                    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img=img.swapaxes(0, 1)
                    
                    pygame.surfarray.blit_array(window_surface, img)
                    
                    button_menu.draw(window_surface)
                    
                    mouse_pos=pygame.mouse.get_pos()
                    for event in pygame.event.get():
                    
                        # 關閉事件1 (按下右上叉叉，離開遊戲)
                        if event.type == pygame.QUIT:
                            On_status=False
                            sys.exit(0)
                            camera.release()
                            
                        # 關閉事件2 (按下ESC 或 Q，離開遊戲)
                        elif event.type == KEYDOWN:
                            if event.key == K_ESCAPE:
                                On_status=False
                                sys.exit(0)
                                camera.release()
                                
                            elif event.key == K_x:
                                Menu=True
                                Run=False
                                Score=False
                                Demo=False
                                
                            elif event.key == K_f:
                                Fullscreen=not Fullscreen
                                if Fullscreen:
                                    screen=pygame.display.set_mode((1280, 720), FULLSCREEN, 32)
                                else:
                                    screen=pygame.display.set_mode((1280, 720), 0, 32)
                        elif event.type == MOUSEBUTTONDOWN:
                            if button_menu.rect.collidepoint(mouse_pos):
                                Menu=True
                                Run=False
                                Score=False
                                Demo=False
                                
                    pygame.display.update()
#############################################################################################        
    
        except (KeyboardInterrupt, SystemExit):
            pygame.quit()
            cv2.destroyAllWindows()
            camera.release()
        
if __name__ == '__main__':    
    main()

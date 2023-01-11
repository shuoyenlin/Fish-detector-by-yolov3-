# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import cv2
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

#trained_weights_final.h5
#model_data/coco_classes.txt
class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo_weights.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        #含有全部類別的txt
        "classes_path": 'model_data/own_classes.txt',
        #測試圖片時，如果抓不到物件，可以試著把score調低，降低門檻。        
        "score" : 0.03,
        "iou" : 0.01,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes
    
    def hisEqulColor(self,img):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        cv2.equalizeHist(channels[0], channels[0])
        cv2.merge(channels, ycrcb)
        cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB, img)
        return img
    
    def custom_blur_demo(self,image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
        dst = cv2.filter2D(image, -1, kernel=kernel)
        return dst
    
    def adjust_gamma(self,image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def contrast_brightness_image(self,src1, a=1.2, g=10):
        h, w, ch = src1.shape#获取shape的数值，height和width、通道
 
        #新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
        src2 = np.zeros([h, w, ch], src1.dtype)
        dst = cv2.addWeighted(src1, a, src2, 1-a, g)#addWeighted函数说明如下
        return dst
    
    #計算速度
    def speed(self,df_old,df_new):
        speed = 0
        if len(df_new) <= len(df_old):
            df_min = df_new
            df_max = df_old
        else:
            df_max = df_new
            df_min = df_old
    
    
        centerx_max = df_max['centerx']
        centery_max = df_max['centery']
        centerx_min = df_min['centerx']
        centery_min = df_min['centery']
    
        mindis_temp =[]
        
        tags = len(df_min)
        for i in range(len(df_min)):
            dis_temp =[]    
            for j in range(len(df_max)):
                v1 = np.array([centerx_max[j],centery_max[j]])
                v2 = np.array([centerx_min[i],centery_min[i]])
                dis = np.linalg.norm(v2-v1)
                dis_temp.append(dis)                
            
            if min(dis_temp) > 10:
                tags -=1
            else:
                mindis_temp.append(min(dis_temp))
                
        speed = np.mean(mindis_temp)
        if np.isnan(speed) == True:
            speed = 0
        return speed , tags 

    #在圖上畫出軌跡
    def trace(self,draw,df1,df2,color):
        if len(df1) <= len(df2):
            df_min = df1
            df_max = df2
        else:
            df_max = df1
            df_min = df2
        
        centerx_max = df_max['centerx']
        centery_max = df_max['centery']
        centerx_min = df_min['centerx']
        centery_min = df_min['centery']
                
        mindis_temp =[]
        for i in range(len(df_min)):
            dis_temp =[]
            for j in range(len(df_max)):
                v1 = np.array([centerx_max[j],centery_max[j]])
                v2 = np.array([centerx_min[i],centery_min[i]])
                dis = np.linalg.norm(v2-v1)
                dis_temp.append(dis)                
                endpoint = np.argmin(dis_temp)
            if min(dis_temp) < 50:
                draw = ImageDraw.Draw(image)
                linestart = (df_min['centerx'][i] , df_min['centery'][i])
                lineend = (df_max['centerx'][endpoint] , df_max['centery'][endpoint])
                draw.line((linestart,lineend),fill = color, width = 3)
                del draw
           
    
#    def draw_circle(self,event,x,y,flags,param):
#        global ix,iy,drawing
#
#            #在圖面上點擊滑鼠左鍵，標記該位置的座標
#        if event == cv2.EVENT_LBUTTONDOWN:
#            drawing = True
#            ix,iy=x,y
#            print(ix,iy)
#                     
#            #当鼠标松开时停止绘图
#        elif event == cv2.EVENT_LBUTTONUP:
#            drawing == False
        
    
    def detect_boundary(self,frame):
        global drawing,ix,iy
        drawing = False
        ix,iy = -1,-1
        def draw_circle(event,x,y,flags,param):
            global ix,iy,drawing,mode

            #在圖面上點擊滑鼠左鍵，標記該位置的座標
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix,iy=x,y
                print(ix,iy)

            #当鼠标松开时停止绘图
            elif event == cv2.EVENT_LBUTTONUP:
                drawing == False
    
        
        
        cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('frame',draw_circle)
        
        a=0
        b=0
        df=pd.DataFrame(columns=['ix','iy'])

        while(1):
            cv2.imshow('frame',frame)
            k=cv2.waitKey(1)
    
        #要關閉程式請按鍵盤上 Q (通常要先按 shift 後再按 Q )
            if k==ord('q'):  
                break

        #將每點擊一次圖的座標位置，儲存至'df'dataframe內  
            if a!=ix and b!=iy:
                if len(df)<=4:
                    a = ix
                    b = iy  
                    new = pd.DataFrame({'ix':a,'iy':b},index=[1])
                    df = df.append(new,ignore_index=True)
                else:
                    break               
        
        df=df.loc[1:]        
        cv2.destroyAllWindows()
        x_list =sorted(df['ix'])
        y_list =sorted(df['iy'])
        print(x_list,y_list)
        return x_list,y_list
        
    
    
    def detect_image(self, image ,detect_img , dfI_old,dfII_old,dfIII_old,x_list):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(detect_img, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (detect_img.width - (detect_img.width % 32),
                              detect_img.height - (detect_img.height % 32))
            boxed_image = letterbox_image(detect_img, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [detect_img.size[1], detect_img.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * detect_img.size[1] + 0.5).astype('int32'))
        thickness = (detect_img.size[0] + detect_img.size[1]) // 600

       
        numI,numII,numIII = 0,0,0
        fishdfI = pd.DataFrame(columns=['label','left','top','right','bottom','centerx','centery'])
        fishdfII,fishdfIII = fishdfI,fishdfI
        colorI,colorII,colorIII = (255,0,0),(0,255,0),(0,0,255)
        boundI, boundII, boundIII,boundIV = x_list[0],x_list[1],x_list[2],x_list[3]
        
        
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{},{}'.format(predicted_class,i)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            
            centerx = (left + right) //2
            centery = (top + bottom) //2
#             print(label, (centerx,centery))
            
            if centerx <= boundII-2 and centerx>boundI:

                fishdfI = fishdfI.append({'label':label,'left':left,'top':top,'right':right,'bottom':bottom,
                                          'centerx':centerx,'centery':centery},ignore_index=True)
                numI += 1
                print(label,'I', (centerx,centery))
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i],outline=(255, 0, 0))
                del draw
                
                if len(fishdfI) <= len(dfI_old):
                    df_min = fishdfI
                    df_max = dfI_old
                else:
                    df_max = fishdfI
                    df_min = dfI_old

                centerx_max = df_max['centerx']
                centery_max = df_max['centery']
                centerx_min = df_min['centerx']
                centery_min = df_min['centery']

                mindis_temp =[]
                for i in range(len(df_min)):
                    dis_temp =[]
                    for j in range(len(df_max)):
                        v1 = np.array([centerx_max[j],centery_max[j]])
                        v2 = np.array([centerx_min[i],centery_min[i]])
                        dis = np.linalg.norm(v2-v1)
                        dis_temp.append(dis)                
                        endpoint = np.argmin(dis_temp)
                    if min(dis_temp) < 15:
                        draw = ImageDraw.Draw(image)
                        linestart = (df_min['centerx'][i] , df_min['centery'][i])
                        lineend = (df_max['centerx'][endpoint] , df_max['centery'][endpoint])
                        draw.line((linestart,lineend),fill = colorI, width = 3)
                        del draw
                    
            elif centerx >= boundII and centerx <= boundIII:
                fishdfII = fishdfII.append({'label':label,'left':left,'top':top,'right':right,'bottom':bottom,
                                          'centerx':centerx,'centery':centery},ignore_index=True)
                numII += 1
                print(label,'II', (centerx,centery))
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i],outline=(0, 255, 0))
                del draw
                
                if len(fishdfII) <= len(dfII_old):
                    df_min = fishdfII
                    df_max = dfII_old
                else:
                    df_max = fishdfII
                    df_min = dfII_old

                centerx_max = df_max['centerx']
                centery_max = df_max['centery']
                centerx_min = df_min['centerx']
                centery_min = df_min['centery']

                mindis_temp =[]
                for i in range(len(df_min)):
                    dis_temp =[]
                    for j in range(len(df_max)):
                        v1 = np.array([centerx_max[j],centery_max[j]])
                        v2 = np.array([centerx_min[i],centery_min[i]])
                        dis = np.linalg.norm(v2-v1)
                        dis_temp.append(dis)                
                        endpoint = np.argmin(dis_temp)
                    if min(dis_temp) < 15:
                        draw = ImageDraw.Draw(image)
                        linestart = (df_min['centerx'][i] , df_min['centery'][i])
                        lineend = (df_max['centerx'][endpoint] , df_max['centery'][endpoint])
                        draw.line((linestart,lineend),fill = colorII, width = 3)
                        del draw                    
                
            elif centerx >= boundIII:
                fishdfIII = fishdfIII.append({'label':label,'left':left,'top':top,'right':right,'bottom':bottom,
                                          'centerx':centerx,'centery':centery},ignore_index=True)
                numIII += 1
                print(label,'III', (centerx,centery))
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i],outline=(0, 0, 255))
                del draw
                
                if len(fishdfIII) <= len(dfIII_old):
                    df_min = fishdfIII
                    df_max = dfIII_old
                else:
                    df_max = fishdfIII
                    df_min = dfIII_old

                centerx_max = df_max['centerx']
                centery_max = df_max['centery']
                centerx_min = df_min['centerx']
                centery_min = df_min['centery']

                mindis_temp =[]
                for i in range(len(df_min)):
                    dis_temp =[]
                    for j in range(len(df_max)):
                        v1 = np.array([centerx_max[j],centery_max[j]])
                        v2 = np.array([centerx_min[i],centery_min[i]])
                        dis = np.linalg.norm(v2-v1)
                        dis_temp.append(dis)                
                        endpoint = np.argmin(dis_temp)
                    if min(dis_temp) < 15:
                        draw = ImageDraw.Draw(image)
                        linestart = (df_min['centerx'][i] , df_min['centery'][i])
                        lineend = (df_max['centerx'][endpoint] , df_max['centery'][endpoint])
                        draw.line((linestart,lineend),fill = colorIII, width = 3)
                        del draw
        

   
                
#             if top - label_size[1] >= 0:
#                 text_origin = np.array([left, top - label_size[1]])
#             else:
#                 text_origin = np.array([left, top + 1])

#             # My kingdom for a good redistributable image drawing library.
#             for i in range(thickness):
#                 draw.rectangle(
#                     [left + i, top + i, right - i, bottom - i],
#                     outline=self.colors[c])
#             draw.rectangle(
#                 [tuple(text_origin), tuple(text_origin + label_size)],
#                 fill=self.colors[c])
#             draw.text(text_origin, label, fill=(0, 0, 0), font=font)
#             del draw
 
        
        
        end = timer()
        
        print(end - start)
        return image ,fishdfI,fishdfII,fishdfIII

    def box_input(self, image,dfI_new,dfII_new,dfIII_new ,dfI_old,dfII_old,dfIII_old):
        start = timer()

        colorI,colorII,colorIII = (255,0,0),(0,255,0),(0,0,255)
        
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        
        print('Write {} boxes for {}'.format((len(dfI_new)+len(dfII_new)+len(dfIII_new)), 'img'))


        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 600

        labelI_list = dfI_new['label'].str.split('_')
        leftI_list = dfI_new['left']
        topI_list =dfI_new['top']
        rightI_list =dfI_new['right']
        bottomI_list =dfI_new['bottom']
        
        labelII_list = dfII_new['label'].str.split('_')
        leftII_list = dfII_new['left']
        topII_list =dfII_new['top']
        rightII_list =dfII_new['right']
        bottomII_list =dfII_new['bottom']
        
        labelIII_list = dfIII_new['label'].str.split('_')
        leftIII_list = dfIII_new['left']
        topIII_list =dfIII_new['top']
        rightIII_list =dfIII_new['right']
        bottomIII_list =dfIII_new['bottom']

        for i in range(len(labelI_list)):
            
            label = '{} '.format(labelI_list[i][0])
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top = topI_list[i]
            left = leftI_list[i]
            bottom = bottomI_list[i]
            right = rightI_list[i]
            
            centerx = (left + right) //2
            centery = (top + bottom) //2
            print(label, (centerx, centery))
            
            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=(255,0,0))
            del draw
            
            if len(dfI_new) <= len(dfI_old):
                df_min = dfI_new
                df_max = dfI_old
            else:
                df_max = dfI_new
                df_min = dfI_old

            centerx_max = df_max['centerx']
            centery_max = df_max['centery']
            centerx_min = df_min['centerx']
            centery_min = df_min['centery']

            mindis_temp =[]
            for i in range(len(df_min)):
                dis_temp =[]
                for j in range(len(df_max)):
                    v1 = np.array([centerx_max[j],centery_max[j]])
                    v2 = np.array([centerx_min[i],centery_min[i]])
                    dis = np.linalg.norm(v2-v1)
                    dis_temp.append(dis)                
                    endpoint = np.argmin(dis_temp)
                if min(dis_temp) < 15:
                    draw = ImageDraw.Draw(image)
                    linestart = (df_min['centerx'][i] , df_min['centery'][i])
                    lineend = (df_max['centerx'][endpoint] , df_max['centery'][endpoint])
                    draw.line((linestart,lineend),fill = colorI, width = 3)
                    del draw
            
            
        for i in range(len(labelII_list)):
            
            label = '{} '.format(labelII_list[i][0])
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top = topII_list[i]
            left = leftII_list[i]
            bottom = bottomII_list[i]
            right = rightII_list[i]
            
            centerx = (left + right) //2
            centery = (top + bottom) //2
            print(label, (centerx, centery))
            
            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=(0,255,0))
            del draw
            
            if len(dfII_new) <= len(dfII_old):
                df_min = dfII_new
                df_max = dfII_old
            else:
                df_max = dfII_new
                df_min = dfII_old

            centerx_max = df_max['centerx']
            centery_max = df_max['centery']
            centerx_min = df_min['centerx']
            centery_min = df_min['centery']

            mindis_temp =[]
            for i in range(len(df_min)):
                dis_temp =[]
                for j in range(len(df_max)):
                    v1 = np.array([centerx_max[j],centery_max[j]])
                    v2 = np.array([centerx_min[i],centery_min[i]])
                    dis = np.linalg.norm(v2-v1)
                    dis_temp.append(dis)                
                    endpoint = np.argmin(dis_temp)
                if min(dis_temp) < 15:
                    draw = ImageDraw.Draw(image)
                    linestart = (df_min['centerx'][i] , df_min['centery'][i])
                    lineend = (df_max['centerx'][endpoint] , df_max['centery'][endpoint])
                    draw.line((linestart,lineend),fill = colorII, width = 3)
                    del draw
            
        for i in range(len(labelIII_list)):
            
            label = '{} '.format(labelIII_list[i][0])
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top = topIII_list[i]
            left = leftIII_list[i]
            bottom = bottomIII_list[i]
            right = rightIII_list[i]
            
            centerx = (left + right) //2
            centery = (top + bottom) //2
            print(label, (centerx, centery))
            
            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=(0,0,255))
            del draw
            
        
            if len(dfIII_new) <= len(dfIII_old):
                df_min = dfIII_new
                df_max = dfIII_old
            else:
                df_max = dfIII_new
                df_min = dfIII_old

            centerx_max = df_max['centerx']
            centery_max = df_max['centery']
            centerx_min = df_min['centerx']
            centery_min = df_min['centery']

            mindis_temp =[]
            for i in range(len(df_min)):
                dis_temp =[]
                for j in range(len(df_max)):
                    v1 = np.array([centerx_max[j],centery_max[j]])
                    v2 = np.array([centerx_min[i],centery_min[i]])
                    dis = np.linalg.norm(v2-v1)
                    dis_temp.append(dis)                
                    endpoint = np.argmin(dis_temp)
                if min(dis_temp) < 15:
                    draw = ImageDraw.Draw(image)
                    linestart = (df_min['centerx'][i] , df_min['centery'][i])
                    lineend = (df_max['centerx'][endpoint] , df_max['centery'][endpoint])
                    draw.line((linestart,lineend),fill = colorIII, width = 3)
                    del draw

            
#             del draw

        end = timer()
        print(end - start)
        return image
    
    
    
    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path="testout.avi"):
    import cv2
    from matplotlib import pyplot as plt
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
#     video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_FourCC    = cv2.VideoWriter_fourcc(*'XVID')
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "testout.avi" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    counts = 0
    dfI_new = pd.DataFrame( columns=['label','left','top','right','bottom','centerx','centery'])
    dfII_new,dfIII_new = dfI_new,dfI_new
    dfI_old,dfII_old,dfIII_old = dfI_new,dfI_new,dfI_new
    df_speed = pd.DataFrame(columns=['speedI','speedII','speedIII','tagI','tagII','tagIII',
                                     'ave_speedI','ave_speedII','ave_speedIII','upsideI','upsideII','upsideIII'])
    while True:
        return_value, frame = vid.read()
        if return_value == False:
            break
        
        if counts == 0:            

            x_list,y_list = yolo.detect_boundary(frame)
            #x_list = [20,256,505,720]
            #y_list = [30,64,250,390]
            
        if counts == 1500:

            print(df_speed)

            yolo.close_session()
            break
        
        if counts%2 == 0:
            
            dfI_old = dfI_new
            dfII_old = dfII_new
            dfIII_old = dfIII_new
            
            image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            detect_img = yolo.contrast_brightness_image(frame)
            detect_img = Image.fromarray(cv2.cvtColor(detect_img,cv2.COLOR_BGR2RGB))
            image,dfI_new,dfII_new,dfIII_new= yolo.detect_image(image,detect_img,dfI_old,dfII_old,dfIII_old,x_list)
            result = np.asarray(image)
            result =cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
            
            speedI , tagI  = yolo.speed(dfI_old,dfI_new)
            speedII , tagII  = yolo.speed(dfII_old,dfII_new)
            speedIII , tagIII  = yolo.speed(dfIII_old,dfIII_new)
            
            upsideI = round(dfI_new['centery'].mean() / y_list[3],2)
            upsideII = round(dfII_new['centery'].mean() / y_list[3],2)
            upsideIII = round(dfIII_new['centery'].mean() / y_list[3],2)
            
            if counts<=40:
                ave_speedI = round(df_speed['speedI'].mean(),2)
                ave_speedII = round(df_speed['speedII'].mean(),2)
                ave_speedIII = round(df_speed['speedIII'].mean(),2)
            else:
                ave_speedI = round(df_speed['speedI'][int(counts/2)-10:int(counts/2)].mean(),2)
                ave_speedII = round(df_speed['speedII'][int(counts/2)-10:int(counts/2)].mean(),2)
                ave_speedIII = round(df_speed['speedIII'][int(counts/2)-10:int(counts/2)].mean(),2)
            
            df_speed = df_speed.append({'speedI':speedI,'speedII':speedII,'speedIII':speedIII,
                                        'tagI':tagI,'tagII':tagII,'tagIII':tagIII,
                                        'ave_speedI':ave_speedI,'ave_speedII':ave_speedII,'ave_speedIII':ave_speedIII,
                                        'upsideI':upsideI,'upsideII':upsideII,'upsideIII':upsideIII},
                                       ignore_index=True)
            
            
            
            print('I區 speed :{} ,隻數 :{} ,平均:{} ,高低指數:{}'.format(speedI,tagI,ave_speedI,upsideI))
            print('II區 speed :{} ,隻數 :{} ,平均:{} ,高低指數:{}'.format(speedII,tagII,ave_speedII,upsideII))
            print('III區 speed :{} ,隻數 :{} ,平均:{} ,高低指數:{}'.format(speedIII,tagIII,ave_speedIII,upsideIII))
        else:
            image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            image = yolo.box_input(image ,dfI_new,dfII_new,dfIII_new ,dfI_old,dfII_old,dfIII_old)
            result = np.asarray(image)
            result =cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
            
        
        
        curr_time = timer()
        counts +=1
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text='{};{};{}'.format(tagI,ave_speedI,upsideI), org=(x_list[0], 450),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.20, color=(0,0,255), thickness=2)
        cv2.putText(result, text='{};{};{}'.format(tagII,ave_speedII,upsideII), org=(x_list[1], 450),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.20, color=(0,255,0), thickness=2)
        cv2.putText(result, text='{};{};{}'.format(tagIII,ave_speedIII,upsideIII), org=(x_list[2], 450),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.20, color=(255,0,0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
#        plt.imshow(image)
#        plt.show
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

    
    

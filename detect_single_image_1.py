import os

from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import numpy as np
import time



config_car_yolov3 = r'D:\mod\configs\yolov3.py'
checkpoint_car_inf_yolov3 = r'D:\mod\weights\my_custom_dataset\yolov3_909.pth'
# build the model from a config file and a checkpoint file
model_car_inf_yolov3 = init_detector(config_car_yolov3, checkpoint_car_inf_yolov3, device='cuda:0')

config_car_detr = r'D:\mod\configs\detr.py'
checkpoint_car_inf_detr = r'D:\mod\weights\my_custom_dataset\detr_912.pth'
# build the model from a config file and a checkpoint file
model_car_inf_detr = init_detector(config_car_detr, checkpoint_car_inf_detr, device='cuda:0')

config_car_mask = r'D:\mod\configs\mask.py'
checkpoint_car_inf_mask = r'D:\mod\weights\my_custom_dataset\mask_895.pth'
# build the model from a config file and a checkpoint file
model_car_inf_mask = init_detector(config_car_mask, checkpoint_car_inf_mask, device='cuda:0')

config_car_faster = r'D:\mod\configs\faster.py'
checkpoint_car_inf_faster = r'D:\mod\weights\my_custom_dataset\faster_908.pth'
# build the model from a config file and a checkpoint file
model_car_inf_faster = init_detector(config_car_faster, checkpoint_car_inf_faster, device='cuda:0')

config_car_libra = r'D:\mod\configs\libra.py'
checkpoint_car_inf_libra = r'D:\mod\weights\my_custom_dataset\libra_880.pth'
# build the model from a config file and a checkpoint file
model_car_inf_libra = init_detector(config_car_libra, checkpoint_car_inf_libra, device='cuda:0')

config_car_retina = r'D:\mod\configs\retina.py'
checkpoint_car_inf_retina = r'D:\mod\weights\my_custom_dataset\retina_930.pth'
# build the model from a config file and a checkpoint file
model_car_inf_retina = init_detector(config_car_retina, checkpoint_car_inf_retina, device='cuda:0')

config_car_yolof = r'D:\mod\configs\yolof.py'
checkpoint_car_inf_yolof = r'D:\mod\weights\my_custom_dataset\yolof_921.pth'
# build the model from a config file and a checkpoint file
model_car_inf_yolof = init_detector(config_car_yolof, checkpoint_car_inf_yolof, device='cuda:0')

config_car_yolox = r'D:\mod\configs\yolox.py'
checkpoint_car_inf_yolox = r'D:\mod\weights\my_custom_dataset\yolox_893.pth'
# build the model from a config file and a checkpoint file
model_car_inf_yolox = init_detector(config_car_yolox, checkpoint_car_inf_yolox, device='cuda:0')

config_car_deformable_detr = r'D:\mod\configs\deformable_detr.py'
checkpoint_car_inf_deformable_detr = r'D:\mod\weights\my_custom_dataset\deformable_detr_928.pth'
# build the model from a config file and a checkpoint file
model_car_inf_deformable_detr = init_detector(config_car_deformable_detr, checkpoint_car_inf_deformable_detr, device='cuda:0')



#这段函数定义了一个detection函数,用于使用MMDetection模型对输入图像进行目标检测,并过滤掉置信度低于设定阈值（0.5）的检测结果。
def detection(img, model):
    result = inference_detector(model, img)
    # print(result)

    score_thres = 0.01

    if len(result) == 2:
        result = list(result)
        for i in range(len(result[0])):
            if result[0][i].size != 0:
                bboxes = np.vstack(result[0][i])
                scores = bboxes[:, -1]
                inds = scores > score_thres
                bboxes = bboxes[inds, :]
                segms = result[1][i]
                filtered_segms = []
                for j, flag in enumerate(inds):
                    if flag:
                        filtered_segms.append(segms[j])
                segms = filtered_segms
                result[0][i] = bboxes
                result[1][i] = segms
        result = tuple(result)
    else:
        for i in range(len(result)):
            if result[i].size != 0:
                bboxes = np.vstack(result[i])
                scores = bboxes[:, -1]
                inds = scores > score_thres
                bboxes = bboxes[inds, :]
                result[i] = bboxes
    time.sleep(0.01)
    model.show_result(img, result, out_file='result.jpg')

    return result

def yolov3_inf(img):
    result = detection(img, model_car_inf_yolov3)
    return result

def detr_inf(img):
    result = detection(img, model_car_inf_detr)
    return result

def mask_inf(img):
    result = detection(img, model_car_inf_mask)
    return result[0]

def faster_inf(img):
    result = detection(img, model_car_inf_faster)
    return result

def libra_inf(img):
    result = detection(img, model_car_inf_libra)
    return result

def retina_inf(img):
    result = detection(img, model_car_inf_retina)
    return result

def yolof_inf(img):
    result = detection(img, model_car_inf_yolof)
    return result

def yolox_inf(img):
    result = detection(img, model_car_inf_yolox)
    return result

def deformable_detr_inf(img):
    result = detection(img, model_car_inf_deformable_detr)
    return result









#img = r'D:\project\ronghe\val_people\FLIR_08864.jpeg'

#img =  r'C:\Users\a\PycharmProjects\pythonProject\cross_modal_attack\path_adv\clean\2.jpg'

#img = r'1.jpg'
#result = yolov3_inf(img)
#print(result)
#print(result[0])
#print(result[0][0])
#print(result[0][0][4])
#print(result[0].shape)


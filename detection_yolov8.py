# from ultralytics import YOLO
#
# # Load a model
# model = YOLO(r"C:\Users\a\Desktop\mmdetection_model\people_infrared\U版Yolov8红外行人检测权重\yolov8_883.pt")  # load a pretrained model (recommended for training)
#
# # Use the model
# # model.train(data="coco128.yaml", epochs=3)  # train the model
# # metrics = model.val()  # evaluate model performance on the validation set
# results = model(r"C:\Users\a\Desktop\FLIR_v1_v2\FLIR_v2_1\images_car_v2\images_car_v2\train\data\video-2SReBn5LtAkL5HMj2-frame-006281-3qo8RBEJPPyt5MANL.jpg")  # predict on an image
# # path = model.export(format="onnx")  # export the model to ONNX format
#
# print('results = ', results)
#
# # print('results.shape = ', results.shape)
# #
# # print('results[0][4] = ', results[0][4])


from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO(r"D:\mod\weights\my_custom_dataset\yolov8_883.pt")

def yolov8_inf(im2):
    results = model.predict(source=im2, save=False, save_txt=False)  # save predictions as labels
    shape = results[0].boxes.shape

    if results[0].boxes.shape == (0, 6):
        x1, y1, x2, y2 = 0, 0, 0, 0
        conf = 100

    else:
        x1, y1, x2, y2 = int(results[0].boxes.xyxy[0][0]), int(results[0].boxes.xyxy[0][1]), int(results[0].boxes.xyxy[0][2]), int(results[0].boxes.xyxy[0][3])
        conf = float(results[0].boxes.conf[0])
    return shape, x1, y1, x2, y2, conf


#shape, x1, y1, x2, y2, conf=yolov8_inf("D:\\project\\ronghe\\images\\FLIR_00922.jpeg")
#print("shape, x1, y1, x2, y2, conf=",shape, x1, y1, x2, y2, conf)
#print(shape[0])
# print('results[0].boxes = ', results[0].boxes)
# print('results[0].boxes.conf = ', results[0].boxes.conf)
# print('results[0].boxes.conf[0] = ', results[0].boxes.conf[0])
# print('results[0].boxes.shape = ', results[0].boxes.shape)
#
# if results[0].boxes.shape == (2, 6):
#     print('yes1')
# else:
#     print('no1')
#
# print('float(results[0].boxes.conf[0]) = ', float(results[0].boxes.conf[0]))
#
# if float(results[0].boxes.conf[0]) == 0.7018035054206848:
#     print('yes2')
# else:
#     print('no2')
#
# print('results[0].boxes.xyxy[0][0] = ', int(results[0].boxes.xyxy[0][0]))




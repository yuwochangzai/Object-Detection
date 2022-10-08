import cv2
#from imread_from_url import imread_from_url
import os
import shutil

from YOLOv7 import YOLOv7

# Initialize YOLOv7 object detector
model_path = "E:/github/machinelearning-samples/samples/csharp/getting-started/DeepLearning_ObjectDetection_Onnx/ObjectDetectionConsoleApp/assets/Model/yolov7/yolov7_736x1280.onnx"
img_folder = 'E:/github/machinelearning-samples/samples/csharp/getting-started/DeepLearning_ObjectDetection_Onnx/ObjectDetectionConsoleApp/assets/images'
yolov7_detector = YOLOv7(model_path, conf_thres=0.3, iou_thres=0.5)

#检查可疑图片  
# boxes坐标格式：[左上角横坐标，左上角纵坐标，右下角横坐标，右下角纵坐标]
def checkDubious(boxes,class_ids):
    count = len(class_ids)
    for p in range(count):
        if class_ids[p]!=0:#非person
            continue
        for c in range(count):
            if class_ids[c]!=2:#非car
                continue
            person = boxes[p]
            car = boxes[c]
            # 取不相交的对立事件
            if not ( person[1] > car[3] or person[0]>car[2] or car[1]>person[3] or car[0]>person[2]) :
                return True
    return False

# Read image
# img_url = "https://www.hujun.site/wp-content/uploads/2022/09/park-1024x473.jpg"
# img = imread_from_url(img_url)

filenames = os.listdir(img_folder)
filenames = [f for f in filenames if f.endswith('.jpg')]
for filename in filenames:
    filepath = os.path.join(img_folder,filename)
    filename=os.path.splitext(filename)[0]
    img = cv2.imread(filepath)

    # Detect Objects
    boxes, scores, class_ids = yolov7_detector(img)

    # Draw detections
    combined_img = yolov7_detector.draw_detections(img)
    #cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    #cv2.imshow("Detected Objects", combined_img)
    cv2.imwrite("doc/img/{}.jpg".format(filename), combined_img)
    print('{}.jpg完成检测'.format(filename))

    #检测是否可疑图片（存在person矩形框与car矩形框相交的情况）
    if checkDubious(boxes,class_ids):
        #cv2.imwrite("doc/img/dubious/{}.jpg".format(filename), combined_img)
        shutil.copy("doc/img/{}.jpg".format(filename),'doc/img/dubious/')
    #cv2.waitKey(0)



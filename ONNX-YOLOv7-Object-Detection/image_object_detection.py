import cv2
#from imread_from_url import imread_from_url
import os

from YOLOv7 import YOLOv7

# Initialize YOLOv7 object detector
model_path = "E:/github/machinelearning-samples/samples/csharp/getting-started/DeepLearning_ObjectDetection_Onnx/ObjectDetectionConsoleApp/assets/Model/yolov7/yolov7_736x1280.onnx"
img_folder = 'E:/github/machinelearning-samples/samples/csharp/getting-started/DeepLearning_ObjectDetection_Onnx/ObjectDetectionConsoleApp/assets/images'
yolov7_detector = YOLOv7(model_path, conf_thres=0.3, iou_thres=0.5)

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
    #cv2.waitKey(0)

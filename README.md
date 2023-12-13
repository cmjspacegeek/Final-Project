## AI Image Detection Model using OpenCV and ImageAI

Welcome to my Pi's and Python Class Final Project! In this repository we present an exciting exploration into the realm of computer vision through the implementation of an AI Image Detection Model using OpenCV and ImageAI. This project leverages the power of OpenCV a popular open-source computer vision library to develop a robust system capable of detecting and analyzing objects within images along with the power of ImageAI to come together to make a powerful and capable Image Detection Program.


#### Project Overview
This project represents a solo exploration into the dynamic intersection of computer science artificial intelligence and image processing. The focus lies on practical applications of image detection showcasing the potential of AI in visual recognition tasks through the integration of OpenCV and ImageAI.

#### List of Objects
- Person  
- Bicycle  
- Car  
- Motorcycle  
- Airplane  
- Bus  
- Train  
- Truck  
- Boat  
- Traffic Light  
- Fire Hydrant  
- Stop Sign  
- Parking Meter  
- Bench  
- Bird  
- Cat  
- Dog  
- Horse  
- Sheep  
- Cow  
- Elephant  
- Bear  
- Zebra  
- Giraffe  
- Backpack  
- Umbrella  
- Handbag  
- Tie  
- Suitcase  
- Frisbee  
- Skis  
- Snowboard  
- Sports Ball  
- Kite  
- Baseball Bat  
- Baseball Glove  
- Skateboard  
- Surfboard  
- Tennis Racket  
- Bottle  
- Wine Glass  
- Cup  
- Fork  
- Knife  
- Spoon  
- Bowl  
- Banana  
- Apple  
- Sandwich  
- Orange  
- Broccoli  
- Carrot  
- Hot Dog  
- Pizza  
- Donuts  
- Cake  
- Chair  
- Couch  
- Potted Plant  
- Bed  
- Dining Table  
- Toilet  
- TV  
- Laptop  
- Mouse  
- Remote  
- Keyboard  
- Cell Phone  
- Microwave  
- Oven  
- Toaster  
- Sink  
- Refrigerator  
- Book  
- Clock  
- Vase  
- Scissors  
- Teddy bear  
- Hair Drier  
- Toothbrush

#### Installation
to install, clone the GitHub repository linked [here](https://github.com/cmjspacegeek/Final-Project.git) and download the dependencies linked [here](https://github.com/OlafenwaMoses/ImageAI/blob/master/requirements.txt) and install using the command `pip install -r requirements.txt` or using this command 
```shell
pip install cython pillow>=7.0.0 numpy>=1.18.1 opencv-python>=4.1.2 torch>=1.9.0 --extra-index-url https://download.pytorch.org/whl/cpu torchvision>=0.10.0 --extra-index-url https://download.pytorch.org/whl/cpu pytest==7.1.3 tqdm==4.64.1 scipy>=1.7.3 matplotlib>=3.4.3 mock==4.0.3
```
Then run the command below to install ImageAI
```  pip install imageai --upgrade ```
Download the YOLOv3 object detection model linked [here](https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/yolov3.pt)
Open the Final-Project File and run project.py
Code for project.py
```python
from imageai.Detection import ObjectDetection  
import os  
# Python program to capture a single image  
# using pygame library  
import sys # to access the system  
import cv2  
# importing the pygame library  
import pygame  
import pygame.camera  
from time import sleep  
try:  
    execution_path = os.getcwd()  
    detector = ObjectDetection()  
    detector.setModelTypeAsYOLOv3()  
    detector.setModelPath( os.path.join(execution_path , "yolov3.pt"))  
    detector.loadModel()  
  
  
  
    # initializing  the camera  
    pygame.camera.init()  
  
    # make the list of all available cameras  
    camlist = pygame.camera.list_cameras()  
  
    # if camera is detected or not  
    if camlist:  
  
        # initializing the cam variable with default camera  
        cam = pygame.camera.Camera(camlist[0], (1120,480))  
  
        # opening the camera  
        cam.start()  
  
        while True:  
            # capturing the single image  
            image = cam.get_image()  
  
            # saving the image  
            pygame.image.save(image, "filename.jpg")  
            detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "filename.jpg"), output_image_path=os.path.join(execution_path , "filename.jpg"), minimum_percentage_probability=30)  
            img = cv2.imread("filename.jpg", cv2.IMREAD_ANYCOLOR)  
            cv2.imshow("AI", img)  
            cv2.waitKey(1)  
  
  
    # if camera is not detected the moving to else part  
    else:  
        print("No camera on current device")  
  
except ValueError:  
    print("\nError Model not Found")
```

You need to run the code from the command line, once you have, one of three things will happen

1. It works and a window will open with a live camera feed, (this could be laggy based on the power of your CPU)

2. An error message that reads `No camera on current device` , this means that the code has not detected a camera. check your OS settings and re-try

3. An error message that reads `Error Model not Found`, this means that the code has not found a Model, try looking in the folder that the code in is 


Thank you for installing and running my code, if you would like to make changes email me at cmjspacgeek@gmail.com

Thank you
-Connor

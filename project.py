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

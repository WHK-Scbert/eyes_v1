#!/usr/bin/python3
import cv2
from datetime import datetime
from time import sleep
import os
import tinys3
import yaml
import datetime as dt
import sys
import subprocess
import os
from picamera2 import Picamera2


# testing
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

# photo props
image_width = cfg['image_settings']['horizontal_res']
image_height = cfg['image_settings']['vertical_res']
file_extension = cfg['image_settings']['file_extension']
file_name = cfg['image_settings']['file_name']
photo_interval = cfg['image_settings']['photo_interval'] # Interval between photo (in seconds)
image_folder = cfg['image_settings']['folder_name']


# s3 props
bucket = cfg['s3']['bucket_name']

BUCKET = "s3://{bucket}/"

DEST = BUCKET + "sub-folder-in-your-bucket-here/"
CURRENT_DATE = dt.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
CURRENT_TIME = dt.datetime.now().strftime('%m%d%Y%H%M%S')


# Grab images as numpy arrays and leave everything else to OpenCV.
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cv2.startWindowThread()

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (image_width, image_height)}))
picam2.start()
count = 0



while True:
    im = picam2.capture_array()
    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    faces = face_detector.detectMultiScale(grey, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0))
    
    if len(faces) != 0:
        count += 1
        out_path = f"/faces/face_{count}.{file_extension}"
        cv2.imwrite(out_path, im)

        # Upload to S3
        SRC_DIR = out_path
        CMD = "s3cmd put --acl-public %s/*.* %s" % (SRC_DIR, DEST)
        subprocess.call(CMD, shell=True)


        # Delete local file
        os.remove(out_path)
        


    cv2.imshow("Camera", im)

    




      



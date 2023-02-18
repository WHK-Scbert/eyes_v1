# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time
import yaml
import cv2
from datetime import datetime
import datetime as dt
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
from picamera2 import Picamera2
import numpy as np
import boto3
import os


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  #configure boto3
  s3 = boto3.resource('s3')
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

  # camera info props
  camera_name = cfg['camera_info']['camera_name']
  camera_location = cfg['camera_info']['location_name']
  camera_type = cfg['camera_info']['camera_type']
  # s3 props
  bucket = cfg['s3']['bucket_name']
  s3_folder = cfg['s3']['folder_name']
  
  # Save image to local folder
  # Continuously capture images from the camera and run inference
  picture_counter = 0

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()
  #Video input
  picam2 = Picamera2()
  picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (width, height)}))
  picam2.start()

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  
  i = 0
  while True:
    if(i % 4 == 0):
        i = i+1
    else:
        image = picam2.capture_array()
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        counter += 1
        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Run object detection estimation using the model.
        detection_result = detector.detect(input_tensor)
        
        
        filtered_detections = [
            detection for detection in detection_result.detections
            if any(category.category_name == "person" and category.score > 0.65 for category in detection.categories)
        ]
        filtered_detection_result = processor.DetectionResult(detections=filtered_detections)
        # Draw keypoints and edges on input image
        image = utils.visualize(image, filtered_detection_result)
        
        #image = utils.visualize(image, detection_result)
        if filtered_detections:
            picture_counter += 1
            #timestamp
            FOLDER_DATE = dt.datetime.now().strftime('%m%d%Y')
            FOLDER_HOUR = dt.datetime.now().strftime('%H00H') #Categorized by hour
            CURRENT_DATE = dt.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
            CURRENT_TIME = dt.datetime.now().strftime('%m%d%Y%H%M%S')

            # Save image to local folder
            out_path = f"faces/face_{picture_counter}.{file_extension}"
            amazon_path = f"{s3_folder}/{FOLDER_DATE}/{FOLDER_HOUR}/face_{picture_counter}.{file_extension}"

            cv2.imwrite(out_path, image)
            # Upload to S3
            s3.meta.client.upload_file(out_path, bucket, amazon_path,
                ExtraArgs={'Metadata': {'Capture_Type': 'face', 
                'Date': CURRENT_DATE , 
                'Time': CURRENT_TIME, 
                'Camera_Location': camera_location, 
                'Camera_Name': camera_name, 
                'Camera_Type': camera_type}}
            )
            
            # Delete local file
            os.remove(out_path)
            
        
        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(fps)
        text_location = (left_margin, row_size)
        cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()

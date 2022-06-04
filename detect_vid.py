#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import time
import numpy as np
import argparse

#Modelos 
from tensor import tensor
from cafe import cafe
from yolo import yolo

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--modelo", 
                    required=False, 
                    help="Selección de modelo: tensor, yolo o cafe", 
                    choices=['tensor', 'yolo', 'cafe'],
                    default='tensor')

parser.add_argument("-f", "--file",
                    type=int, 
                    help="Número de video a procesar", 
                    choices=[1,2,3],
                    default=1)

args = parser.parse_args()

cap = cv2.VideoCapture(f'input/video_{args.file}.mp4')

# get the video frames' width and height for proper saving of videos
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# create the `VideoWriter()` object
out = cv2.VideoWriter(f'output/video_result_{args.file}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# detect objects in each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        image = frame
        result = eval(args.modelo +"(image)")
        cv2.imshow('image', result)
        out.write(result)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

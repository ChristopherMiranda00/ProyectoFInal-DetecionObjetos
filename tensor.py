import cv2
import numpy as np
from preprocess import equalizada

#Cargar Modelos
""" ---------------------------------------------------------------- """
with open('files/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

# get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# load the DNN model
mobile_net_model = cv2.dnn.readNet(model='files/frozen_inference_graph.pb',
                        config='files/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                        framework='TensorFlow')

def tensor(image): 
    #image = equalizada(image)

    image_height, image_width, _ = image.shape
    # create blob from image
    blob = cv2.dnn.blobFromImage(image=image, size=(300,300), mean=(104, 117, 123))
    # create blob from image
    mobile_net_model.setInput(blob)
    # forward pass through the mobile_net_model to carry out the detection
    output = mobile_net_model.forward()

    # loop over each of the detection
    for detection in output[0, 0, :, :]:
        # extract the confidence of the detection
        confidence = detection[2]
        # draw bounding boxes only if the detection confidence is above...
        # ... a certain threshold, else skip
        if confidence > .3:
            # get the class id
            class_id = detection[1]
            # map the class id to the class
            class_name = class_names[int(class_id)-1]
            color = COLORS[int(class_id)]
            # get the bounding box coordinates
            box_x = detection[3] * image_width
            box_y = detection[4] * image_height
            # get the bounding box width and height
            box_width = detection[5] * image_width
            box_height = detection[6] * image_height
            # draw a rectangle around each detected object
            cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
            # put the FPS text on top of the frame
            label = class_name + "  " + str(round(confidence*100 , 2)) #El tipo de objeto y el % confidence de cada objeto
            cv2.putText(image, label, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

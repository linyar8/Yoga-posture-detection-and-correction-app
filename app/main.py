import math
import cv2

import pandas as pd
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

#import mediapipe as mp

offset = 5

def importData():
    dataset={'name':[],'left_elbow_angle':[],'right_elbow_angle':[],
         'left_shoulder_angle':[],'right_shoulder_angle':[],
        'left_knee_angle':[],'right_knee_angle':[]}
    dfv = pd.read_csv('src/main/res/dataset.csv')
    for index,row in dfv.iterrows():
        dataset["name"].append(row['name'])
        dataset["left_elbow_angle"].append(row['left_elbow_angle'])
        dataset["right_elbow_angle"].append(row['right_elbow_angle'])
        dataset["left_shoulder_angle"].append(row['left_shoulder_angle'])
        dataset["right_shoulder_angle"].append(row['right_shoulder_angle'])
        dataset["left_knee_angle"].append(row['left_knee_angle'])
        dataset["right_knee_angle"].append(row['right_knee_angle'])
    return dataset

def detectPose(image, results, display=True):

    # Create a copy of the input image.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    #imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    #results = pose.process(imageRGB)

    # Retrieve the height and width of the input image.
    height, width, _ = image.shape

    # Initialize a list to store the detected landmarks.
    landmarks = []

    # Check if any landmarks are detected.
    if results.pose_landmarks:

        # Draw Pose landmarks on the output image.
        #mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,connections=mp_pose.POSE_CONNECTIONS)
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))

    return output_image, landmarks


def calculateAngle(landmark1, landmark2, landmark3):
    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    while angle < 0:
        # Add 360 to the found angle.
        angle += 180
    if angle > 180:
        angle -= 180

    if(angle > 165):
        angle =180 - angle
    if (angle < -165):
        angle = angle - 180
    if (angle <= 10 and angle >= 0):
        angle += 170

    # Return the calculated angle.
    return math.ceil(angle)

def accuracy(dat,i):
    ans = {}
    maxx = 0
    index =-1
    Count = {}
    final_output = ""
    for j in range(i):
        test=0
        count =0
        avg = 0.0
        for y in dataset.values():
            if (test > 0):
                avg += abs((dat[test-1]-y[j])/y[j])
                if(abs(dat[test-1]-y[j])<20):
                    count+=1
            test+=1
        avg= abs(avg)*100
        if(avg<200 and count >=4):
            ans.update({j:abs(avg)})
            Count.update({j:count})
        for name,cnt in Count.items():
            if cnt>maxx:
                maxx = cnt
                index= name
    Count = sorted(Count.items(),reverse=True)
    max_cnt=0
    for k in Count:
        if(k[1]==maxx):
            max_cnt+=1
        else:
            ans.pop(k[0])
    if(max_cnt==1):
        final_output=list(dataset.values())[0][index]
    if(max_cnt>1):
        m = sorted(list(ans.values()),reverse=True)[0]
        for x,y in ans.items():
            if(m==y):
                final_output=list(dataset.values())[0][x]
                break
    if(len(Count)<1):
        final_output='Unknown Pose'
    return final_output

def classifyPose(landmarks, output_image, display=False):
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)


    left_elbow_angle = calculateAngle(landmarks[11],
                                      landmarks[13],
                                      landmarks[15])

    right_elbow_angle = calculateAngle(landmarks[12],
                                       landmarks[14],
                                       landmarks[16])


    left_shoulder_angle = calculateAngle(landmarks[13],
                                         landmarks[11],
                                         landmarks[23])

    right_shoulder_angle = calculateAngle(landmarks[24],
                                          landmarks[12],
                                          landmarks[14])

    # Get the angle between the left hip, knee and ankle points.
    left_knee_angle = calculateAngle(landmarks[23],
                                     landmarks[25],
                                     landmarks[27])

    # Get the angle between the right hip, knee and ankle points
    right_knee_angle = calculateAngle(landmarks[24],
                                      landmarks[26],
                                      landmarks[28])


    dat = [left_elbow_angle,right_elbow_angle,
           left_shoulder_angle,right_shoulder_angle,
           left_knee_angle,right_knee_angle]
    i = 0
    for x in dataset.get('name'):
        i+=1

    acc = accuracy(dat,i)
    label = acc

    # ----------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------

    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)

        # Write the label on the output image.
    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    # Check if the resultant image is specified to be displayed.
    if display:

        # Display the resultant image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

    else:

        # Return the output image and the classified label.
        return output_image, label


 # Initializing mediapipe pose class.
#mp_pose = mp.solutions.pose

# Initializing mediapipe drawing class, useful for annotation.
#mp_drawing = mp.solutions.drawing_utils

dataset = importData()

def main(frame,results):
    # Setup Pose function for video.
    #pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)

    # Get the width and height of the frame
    frame_height, frame_width, _ = frame.shape

    # Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    # Perform Pose landmark detection.
    frame, landmarks = detectPose(frame, results, display=False)

    # Check if the landmarks are detected.
    if landmarks:
        # Perform the Pose Classification.

        frame, _ = classifyPose(landmarks, frame, display=False)
    return frame
    
# Imports
from niryo_one_tcp_client import *
import time
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import numpy as np
import requests
from io import BytesIO
import base64
import spacy
from openai import OpenAI
import argparse
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import time
import json
import keyboard
import speech_recognition as sr
import pyttsx3

# Default position after calibration
default_x = 0.077
default_y = 0.007
default_z = 0.159

# Data points for roll offsets
X_roll = np.array([
    [0.095, -0.238, np.sqrt((0.095-default_x)**2 + (-0.238-default_y)**2 + (0.18-default_z)**2)],
    [0.338, -0.202, np.sqrt((0.338-default_x)**2 + (-0.202-default_y)**2 + (0.18-default_z)**2)],
    [0.331, 0.171, np.sqrt((0.331-default_x)**2 + (0.171-default_y)**2 + (0.18-default_z)**2)],
    [0.1, 0.162, np.sqrt((0.1-default_x)**2 + (0.162-default_y)**2 + (0.18-default_z)**2)]
])
y_roll = np.array([0.8, 0.6, 0.15, 0.7])

# Data points for pitch offsets
X_pitch = np.array([
    [0.095, -0.238, np.sqrt((0.095-default_x)**2 + (-0.238-default_y)**2 + (0.18-default_z)**2)],
    [0.338, -0.202, np.sqrt((0.338-default_x)**2 + (-0.202-default_y)**2 + (0.18-default_z)**2)],
    [0.331, 0.171, np.sqrt((0.331-default_x)**2 + (0.171-default_y)**2 + (0.18-default_z)**2)],
    [0.1, 0.162, np.sqrt((0.1-default_x)**2 + (0.162-default_y)**2 + (0.18-default_z)**2)]
])
y_pitch = np.array([0.35, 0.15, 0.35, 0.45])

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)

# Linear regression for roll offset with polynomial features
model_roll = make_pipeline(poly, LinearRegression())
model_roll.fit(X_roll, y_roll)

# Linear regression for pitch offset with polynomial features
model_pitch = make_pipeline(poly, LinearRegression())
model_pitch.fit(X_pitch, y_pitch)

# Function to calculate roll, pitch, yaw from x, y, z coordinates with offsets
def calculate_rpy_with_offsets(x, y, z):
    roll = math.atan2(y, z)
    pitch = math.atan2(z, math.sqrt(x**2 + y**2))
    yaw = math.atan2(y, x)
    
    # Calculate distance from the default position
    distance_from_default = np.sqrt((x-default_x)**2 + (y-default_y)**2 + (z-default_z)**2)
    
    # Predict offsets
    roll_offset = model_roll.predict([[x, y, distance_from_default]])[0]
    pitch_offset = model_pitch.predict([[x, y, distance_from_default]])[0]
    
    return roll + roll_offset, pitch + pitch_offset, yaw

#Connect to LLM
client = OpenAI()
nlp = spacy.load("en_core_web_sm")


# Connecting to robot
niryo_one_client = NiryoOneClient()
niryo_one_client.connect("192.168.207.171")  # Replace by robot IP address
gripper_used = RobotTool.GRIPPER_3  # Tool used for picking
gripper_speed = 400
# Load YOLOv8 model
#model = YOLO('yolov8n.pt')

# Define font, color, and circle parameters
#font = cv2.FONT_HERSHEY_SIMPLEX
#color = (0, 255, 0)  # Green color for annotations
#circle_radius = 5
#circle_thickness = -1  # Negative value for filled circle
#
# Initialize RealSense pipeline
pipe = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 6)
cfg.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6)

# Start the pipeline
profile = pipe.start(cfg)

# Get the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Align depth frame to color frame
align_to = rs.stream.color
align = rs.align(align_to)

def generate_response(system_role, prompt, image_data, convo_hist):
    # Add the system role to the conversation history
    convo_hist.append({"role": "system", "content": system_role})
    convo_hist.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{image_data}",
                            "detail":"high"
                        },
                    },
                ],
            })
    
    try:
        # Create the request to the API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=convo_hist,
            max_tokens=300,
        )
        
        # Append assistant's response to the conversation history
        convo_hist.append({"role": "assistant", "content": response.choices[0].message.content})
        
        return response.choices[0].message.content, convo_hist
    
    except Exception as e:
        # Handle exceptions (e.g., API errors)
        print(f"Error: {e}")
        return None, convo_hist
    
def capture_image(color_image):
    

    # Save the captured frame to the specified file path
    file_path = "image.jpg"
    cv2.imwrite(file_path, color_image)
    print("Image captured and saved successfully!")

    return file_path
    
def image(prompt,depth_frame,color_image):
    # Capture the image and get file path and depth frame
    convo_hist = []
    image_path = capture_image(color_image)

    # Convert the image to Base64
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    image_data_base64 = base64.b64encode(image_data).decode('utf-8')

    while True:
        words_to_check = ["give", "take", "move", "retrieve"]
        word_present = False

        for word in words_to_check:
            if word in prompt.lower():
                system_role = "you are an object detector, if object described in prompt is in the 424 * 240 image, please return the top left and bottom right x and y of the object with the origin at the top left in a json format:\n{\n\"x1\": <x-coordinate>,\"y1\": <y-coordinate>}","x2\": <x-coordinate>,\"y2\": <y-coordinate>}. Where x1 and y1 is the top left coordinate and x2 and y2 is the bottom right coordinate please do not include the string json in the output"
                word_present = True
                break
        else:
            system_role = "you are an object detector, please respond to the prompts according to the image provided."

        response, convo_hist = generate_response(system_role, prompt, image_data_base64, convo_hist)
        print("\nGPT-4o :", response)
        
        #if prompt doesn't contain the specified words, continue the chat
        if word_present == False:
            #SpeakText(response)
            prompt = SpeechListener()
        else:
            jsonparsed = json.loads(response)
            return jsonparsed
        
        # Find the target keyword
        # found, result = find_object(response, image_path, depth_frame)
        # print("\nResult:", result)

        # if found:
            # break

        # # Parse the response to give feedback to the user
        # print("\nGPT-4 couldn't find the object. Please refine your prompt with more specific details or different wording.")

        # # Update prompt for the next iteration
        # prompt = input("Please refine your prompt: ")


def SpeechListener():
    #Initialize recognizer
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
        
            #Adjust energy threshold
            print("Adjusting threshold...")
            r.adjust_for_ambient_noise(source, duration=1)
            print("Ready to listen...")
            print("\n Press space to speak")
            
            keyboard.wait('space')
            print("Listening for 5 seconds...")
            audio = r.listen(source, timeout=5,phrase_time_limit=5)
            
            print("Recognizing Speech....")
            prompt = r.recognize_google(audio)
            result = prompt.lower()
            
            print("recognized speech: ", result)
            return result
        
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
    
    except sr.UnknownValueError:
        print("Unknown error occured")

def SpeakText(text):
    
    #Initializer Engine
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    


import numpy as np
import pyrealsense2 as rs

def get_coordinates(pipe, align):
    print("get coordinates")
    
    # Capture frames
    frames = pipe.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("Failed to capture frames.")
        return None

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Take prompt from user
    prompt = SpeechListener()
    
    # Object detection using GPT-4
    gpt_output = image(prompt, depth_frame, color_image)
    if not gpt_output:
        print("Object detection failed.")
        return None

    print(gpt_output)

    # Extract the coordinates from GPT output
    x, y = (gpt_output['x1']+gpt_output['x2'])//2, (gpt_output['y1']+gpt_output['y2'])//2
    x1,y1,x2,y2 = gpt_output['x1'],gpt_output['y1'],gpt_output['x2'],gpt_output['y2']
    
    color = (0, 255, 0) # Green color in BGR
    thickness = 2 # Thickness of the bounding box
    cv2.rectangle(color_image, (x1,y1), (x2, y2), color, thickness)
    cv2.circle(color_image, (x,y), thickness, color,-1)
    cv2.imshow('Image with Bounding Box', color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    # Get depth at the specified coordinates
    depth = depth_frame.get_distance(x, y)
    if depth == 0:
        print("Invalid depth data at the given coordinates.")
        return None

    # Calculate real-world coordinates using intrinsic parameters
    depth_point = rs.rs2_deproject_pixel_to_point(
        depth_frame.profile.as_video_stream_profile().intrinsics, [x, y], depth)
    
    print(f"Real-world coordinates: {depth_point[2]}, {depth_point[0]}, {depth_point[1]}")
    return (depth_point[2], depth_point[0], depth_point[1])

# Example usage:
# pipe and align should be initialized RealSense pipeline and align objects
# get_coordinates(pipe, align)

    
        
        
        
        
        # results = model(color_image)
        # print(results)
        # for result in results:
            # boxes = result.boxes
            # for box in boxes:
                # center_x, center_y = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2), int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                # confidence = box.conf[0]
                # class_id = int(box.cls[0])
                # label = model.names[class_id]

                # # Check if the detected object's confidence is above a threshold
                # if label == targetObject and confidence > 0.2:
                    # # Get the depth at the center of the bounding box
                    # depth = depth_frame.get_distance(center_x, center_y)
                    # if depth > 0:
                        # # Convert depth to real world coordinates
                        # depth_point = rs.rs2_deproject_pixel_to_point(
                            # depth_frame.profile.as_video_stream_profile().intrinsics, [center_x, center_y], depth)
                        #print(depth_point[2], depth_point[0], depth_point[1])
                        #return (depth_point[2], depth_point[0], depth_point[1])  # Return as (x, y, z)
        
        # Check if the timeout of 5 seconds has been reached
        #if time.time() - start_time > 15:
        #    break

   


def move_to(x, y, z, roll, pitch, yaw):
    niryo_one_client.open_gripper(gripper_used, gripper_speed)
    time.sleep(2)
    # add the offset to y and z
    niryo_one_client.move_pose(x_pos=x, y_pos=y, z_pos=z, roll_rot=roll, pitch_rot=pitch, yaw_rot=yaw)
    time.sleep(2)
    niryo_one_client.close_gripper(gripper_used, gripper_speed)
    time.sleep(2)
    niryo_one_client.move_pose(x_pos=0.077, y_pos=0.007, z_pos=0.159, roll_rot=-0.116, pitch_rot=1.213, yaw_rot=0.079)
    

# Trying to calibrate
print("before cali")
status, data = niryo_one_client.calibrate(CalibrateMode.AUTO)
if status is False:
    print("Error: " + data)
print("before pose")
# Getting pose
status, data = niryo_one_client.get_pose()
initial_pose = None
if status is True:
    initial_pose = data
else:
    print("Error: " + data)

# Use coordinates from camera to move robot arm
print("before coords")
coords = get_coordinates(pipe,align)
if coords:
    x, y, z = coords
    y=0-y
    y-=0.1
    z+=0.05
    x-=0.05
   
    # Y negative are diff direction
    roll, pitch, yaw = calculate_rpy_with_offsets(x, y, z)
    print(f"Moving to coordinates: X={x:.2f}, Y={y:.2f}, Z={z:.2f}, roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}")
    move_to(x, y, z,roll, pitch, yaw)
    
else:
    print("No valid coordinates detected.")

# Reset to front position
# niryo_one_client.move_pose(x_pos=0.077, y_pos=0.007, z_pos=0.159, roll_rot=-0.116, pitch_rot=1.213, yaw_rot=0.079)

# Turning learning mode ON
status, data = niryo_one_client.set_learning_mode(True)
if status is False:
    print("Error: " + data)

niryo_one_client.quit()

# Stop the pipeline
pipe.stop()


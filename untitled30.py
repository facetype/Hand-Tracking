import cv2
import os
import mediapipe as mp
import math
import pygame # Will later add physics and a ball that spawns which you can balance on the line drawn with your fingers
import time


mpHands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks

hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)



cap = cv2.VideoCapture(0)

    #for calculating FPS
old_frame = 0
new_frame = 0



    #Testing if averaging out FPS calculation improves FPS
frame_counter = 0
start_time = time.time()
fps = 0
fps = str(fps)


    #Check if the camera opens correctly
if not cap.isOpened():
    print("Error opening camera")
    exit()




while (1):
    #capture frame by frame, cam.read() returns two values, ret and frame
    #ret is a boolean
    #frame is the current frame
    ret, frame = cap.read()  
    frame = cv2.flip(frame,1)
    
    #If frame isnt read correctly, throw error
    if not ret:
        print("Error reading frame")
        break
    
    
    
    
    #Convert the frame from the default BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    
        #FPS, calculating FPS by taking 1 divided by the time since the last frame. for example 1/0.05 = 20 fps
    #new_frame = time.time()
    #fps = 1/(new_frame - old_frame)
    #old_frame = new_frame
    #fps = int(fps)
    #fps = str(fps)
    #cv2.putText(frame, f"FPS {fps} ", (1750,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    
        #Updating FPS every second instead to try to optimize FPS
    frame_counter += 1
    elapsed_time = time.time() - start_time 
        #Updating FPS every second instead of every frame
    if elapsed_time >= 1.0:
        fps = frame_counter / elapsed_time
        fps = int(fps)
        fps = str(fps)
        
        
            #Reset counter and time
        frame_counter = 0
        start_time = time.time()
        
    cv2.putText(frame, f"FPS {fps} ", (1750,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    
    
    #Getting the x,y,z coordinates of the plane, 
    h, w, c = frame.shape
        
    
      #See if any hands are detected
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        
            #Draw landmarks on the frame
        hand_landmarks_1 = results.multi_hand_landmarks[0]
        hand_landmarks_2 = results.multi_hand_landmarks[1]
        mp_drawing.draw_landmarks(frame, hand_landmarks_1, mpHands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, hand_landmarks_2, mpHands.HAND_CONNECTIONS)
        
        
        
            #Calculating pixel coordinates
        index_finger_x = int(hand_landmarks_1.landmark[8].x*w)
        index_finger_y = int(hand_landmarks_1.landmark[8].y*h)
        middle_finger_x = int(hand_landmarks_2.landmark[8].x*w)
        middle_finger_y = int(hand_landmarks_2.landmark[8].y*h)
        
        thumb_1_x = int(hand_landmarks_1.landmark[4].x*w)
        thumb_1_y = int(hand_landmarks_1.landmark[4].y*h)
        thumb_2_x = int(hand_landmarks_2.landmark[4].x*w)
        thumb_2_y = int(hand_landmarks_2.landmark[4].y*h)
        
            #Distance between thumbs on the X-axis
        distance = abs(thumb_1_x - thumb_2_x)
        
            #If thumbs are together, draw a triangle
        if distance < 50:
            cv2.line(frame, (index_finger_x, index_finger_y), (thumb_1_x, thumb_1_y), (0,0,255), 5)
            cv2.line(frame, (middle_finger_x, middle_finger_y), (thumb_2_x, thumb_2_y), (0,0,255), 5)
        
            
            #Calculating the Euclidean distance
        distance = math.sqrt((middle_finger_x - index_finger_x)**2 + (middle_finger_y - index_finger_y)**2)
            
            
            #Find x-y coordinates to draw the distance on
        x_kor = abs(index_finger_x - middle_finger_x)
        y_kor = abs(index_finger_y - middle_finger_y)
        
        x_kor = index_finger_x - x_kor
        y_kor = index_finger_y - y_kor
            
            
            #Displaying the distance
        cv2.putText(frame, f"Distance: {distance:.2f} pixels", (index_finger_x + 10, index_finger_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Distance: {distance:.2f} pixels", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.line(frame, (index_finger_x, index_finger_y), (middle_finger_x, middle_finger_y), (0,0,255),5)
        
        
        
        
        
    

    
    # Display the processed frame
    cv2.imshow('Hand Tracking', frame)

    
    
    #Exit, cv2.waitKey(1) returns a 32-bit int, but the keypress is usually represented in the last
    #8 bits, so 0xFF extracts the last 8 bits for comparison
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#Release the capture and close the window

cap.release()
cv2.destroyAllWindows()









import cv2
import os
import mediapipe as mp
import math


mpHands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks

hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)



cap = cv2.VideoCapture(0)


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
            
            #Calculating the Euclidean distance
        distance = math.sqrt((middle_finger_x - index_finger_x)**2 + (middle_finger_y - index_finger_y)**2)
            
            #Displaying the distance
        cv2.putText(frame, f"Distance: {distance:.2f} pixels", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.line(frame, (index_finger_x, index_finger_y), (middle_finger_x, middle_finger_y), (0,0,255),5)
        
        
        
            
        
    

    
    # Display the processed frame
    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
    #Exit, cv2.waitKey(1) returns a 32-bit int, but the keypress is usually represented in the last
    #8 bits, so 0xFF extracts the last 8 bits for comparison
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#Release the capture and close the window

cap.release()
cv2.destroyAllWindows()









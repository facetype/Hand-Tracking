import cv2
import os
import mediapipe as mp

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
    
    #See if any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
        
    

    
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









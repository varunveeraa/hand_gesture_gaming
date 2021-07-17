import cv2
import pickle
import mediapipe as mp
import pandas as pd
import numpy as np
import pydirectinput
import time


mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

with open('gesture-final.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        # Make Detections
        results = holistic.process(image)
        
        cv2.rectangle(image,(610,400),(435,200),(0,0,225),2)
        cv2.rectangle(image,(35,400),(210,200),(0,0,225),2)
        
        # Recolor image back to BGR for rendering
        
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        
        try:
            # Extract Pose landmarks
            rHand = results.right_hand_landmarks.landmark
            rHand_row = list([[landmark.x, landmark.y, landmark.z] for landmark in rHand])
            
            # Extract Face landmarks
            lHand = results.left_hand_landmarks.landmark
            lHand_row = list([[landmark.x, landmark.y, landmark.z] for landmark in lHand])
            
            # Concate rows
            tempRow = rHand_row + lHand_row
            
            row = sum(tempRow, [])
         
            # Make Detections
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            print(body_language_class, body_language_prob)
            
            # Display Class
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            if body_language_class == 'forward':
                pydirectinput.keyDown('w')
                
            elif body_language_class == 'right':
                pydirectinput.keyDown('d')
                time.sleep(0.5)
                pydirectinput.keyUp('d')
                
            
            elif body_language_class == 'left':
                pydirectinput.keyDown('a')
                time.sleep(0.5)
                pydirectinput.keyUp('a')
            
            elif body_language_class == 'forward-right':
                pydirectinput.keyDown('w')
                pydirectinput.keyDown('d')
                time.sleep(0.30)
                pydirectinput.keyUp('w')
                pydirectinput.keyUp('d')
            
            elif body_language_class == 'forward-left':
                pydirectinput.keyDown('w')
                pydirectinput.keyDown('a')
                time.sleep(0.30)
                pydirectinput.keyUp('w')
                pydirectinput.keyUp('a')
                
            elif body_language_class == 'stop':
                pydirectinput.keyDown('space')
                time.sleep(2)
                pydirectinput.keyUp('space')
            
            elif body_language_class == 'back':
                pydirectinput.keyUp('w')
                pydirectinput.keyDown('s')
                time.sleep(0.5)
                pydirectinput.keyUp('s')
            elif body_language_class == 'drift-right':
                pydirectinput.press('w','a','space')
            
            elif body_language_class == 'drift-left':
                pydirectinput.press('w','d','space')
            
            elif body_language_class == 'boost':
                pydirectinput.press('shift')
            
        except:
            pass
                        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
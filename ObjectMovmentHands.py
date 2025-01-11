import mediapipe as mp
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


xrec, yrec = 200, 200

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        
            x8 = hand_landmarks.landmark[8].x
            y8 = hand_landmarks.landmark[8].y
            x12 = hand_landmarks.landmark[12].x
            y12 = hand_landmarks.landmark[12].y
            w, h, _ = img.shape
            
            
            N_x8 = int(x8 * w)
            N_y8 = int(y8 * h)
            N_x12 = int(x12 * w)
            N_y12 = int(y12 * h)

           
            cv2.circle(img, (N_x8, N_y8), 3, (0, 255, 0), 3)  
            cv2.circle(img, (N_x12, N_y12), 3, (0, 0, 255), 3)  

          
            distance = np.sqrt((x12 - x8)**2 + (y12 - y8)**2)


            if distance <= 0.03:
                xrec = N_x8  
                yrec = N_y8  
                
                
            cv2.rectangle(img, (xrec, yrec), (xrec + 100, yrec + 100), (255, 0, 0), -1)


              

   
    cv2.imshow("Object Movement", img)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

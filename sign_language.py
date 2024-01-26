import cv2
import mediapipe as mp

capture= cv2.VideoCapture(0)

mp_hands= mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils

hands=mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5)
tipIds=[4,8,12,16,20]

def countFingers(image,hand_landmarks,handNumber=0):
    if hand_landmarks:
        #get all the landmarks of the hand visible
        landmarks= hand_landmarks[handNumber].landmark
        finger=[]

        for lm_index in tipIds:
            fingertipY=landmarks[lm_index].y
            fingerBottomY= landmarks[lm_index-2].y 

            thumbTipX= landmarks[lm_index].x 
            thumbBottomX= landmarks[lm_index-2].x 

            if lm_index!=4:
                if fingertipY<fingerBottomY:
                  finger.append(1)
                  print(f"finger with id {lm_index}is open")

                if fingertipY>fingerBottomY:
                    finger.append(0)  
                    print(f"finger with id{lm_index}is close")

            else:
                if thumbTipX<thumbBottomX:     
                    finger.append(0)   
                    print(f"thumb is close")
                if thumbTipX>thumbBottomX:
                    finger.append(1)
                    print(f"thumb is open")


        total_fingers=finger.count(1)  
        text=f"fingers {total_fingers}"      
        cv2.putText(image,text,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
def drawHandLandmarks(image,hand_landmarks)  :
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(image,landmarks,mp_hands.HAND_CONNECTIONS)     

while True:
    success,image= capture.read()
    image=cv2.flip(image,1)
    result= hands.process(image)
    hand_landmarks= result.multi_hand_landmarks
    drawHandLandmarks(image,hand_landmarks)
    countFingers(image,hand_landmarks)
    cv2.imshow("webcam",image)
    key= cv2.waitKey(1)
    if key==32:
        break
cv2.destroyAllWindows()    

            
        




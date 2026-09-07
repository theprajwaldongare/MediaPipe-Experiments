import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

hands=mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    success,img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hands_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img,hands_landmarks,mp_hands.HAND_CONNECTIONS)

            h,w,c = img.shape

            indexTip = hands_landmarks.landmark[8]
            cx, cy = int(indexTip.x * w), int(indexTip.y * h)
            cv2.circle(img, (cx, cy), 15, (100, 0, 255), cv2.FILLED)

            indexDip = hands_landmarks.landmark[7]
            dcx, dcy = int(indexDip.x * w), int(indexDip.y * h)
            cv2.circle(img, (dcx, dcy), 15, (100, 0, 255), cv2.FILLED)

            indexPip = hands_landmarks.landmark[6]
            pcx, pcy = int(indexPip.x * w), int(indexPip.y * h)
            cv2.circle(img, (pcx, pcy), 15, (100, 0, 255), cv2.FILLED)

            indexMcp = hands_landmarks.landmark[5]
            mcx, mcy = int(indexMcp.x * w), int(indexMcp.y * h)
            cv2.circle(img, (mcx, mcy), 15, (100, 0, 255), cv2.FILLED)

            # y condition satisfy when the finger is above horizontal level .. 
            # but still finger is not up as x are not vertically aligned ..so we check tip and mcp x diff .. it gives x are vertically aligned not only sideway(only condition check of y case)
            xDiff = abs(cx-mcx) 

            if cy<dcy and dcy<pcy and pcy<mcy and xDiff<50:
                cv2.putText(img,"Index is UP!!",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(79, 78, 10),2)


            # for id,lm in enumerate(hands_landmarks.landmark):
            #     h,w,c = img.shape

            #     cx,cy=int(lm.x*w),int(lm.y*h)

            #     if id==8 or id==7 or id==6 or id==5:
            #         cv2.circle(img,(cx,cy),15,(255, 0, 255),cv2.FILLED)
            #     if id==12 or id==11 or id==10 or id==9:
            #         cv2.circle(img,(cx,cy),15,(0, 255, 255),cv2.FILLED)




    # cv2.putText(img,"Hello Mediapipe",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)
    cv2.imshow("MediaPipe Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


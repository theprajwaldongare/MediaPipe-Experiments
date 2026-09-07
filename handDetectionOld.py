import cv2
import pyautogui
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


            # for id,lm in enumerate(hands_landmarks.landmark):
            #     h,w,c = img.shape

            #     cx,cy=int(lm.x*w),int(lm.y*h)

            #     if id==8:
            #         cv2.circle(img,(cx,cy),15,(255, 0, 255),cv2.FILLED)


    cv2.imshow("MediaPipe Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


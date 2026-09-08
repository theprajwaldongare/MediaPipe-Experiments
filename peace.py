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

overlay_size = 150
peaceImage = cv2.imread("assets/peace.jpg")
peaceImage = cv2.resize(peaceImage, (overlay_size, overlay_size))

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
            cv2.circle(img, (cx, cy), 10, (100, 0, 255), cv2.FILLED)

            indexDip = hands_landmarks.landmark[7]
            dcx, dcy = int(indexDip.x * w), int(indexDip.y * h)
            cv2.circle(img, (dcx, dcy), 10, (100, 0, 255), cv2.FILLED)

            indexPip = hands_landmarks.landmark[6]
            pcx, pcy = int(indexPip.x * w), int(indexPip.y * h)
            cv2.circle(img, (pcx, pcy), 10, (100, 0, 255), cv2.FILLED)

            indexMcp = hands_landmarks.landmark[5]
            mcx, mcy = int(indexMcp.x * w), int(indexMcp.y * h)
            cv2.circle(img, (mcx, mcy), 10, (100, 0, 255), cv2.FILLED)
            

            midTip = hands_landmarks.landmark[12]
            MidTx, MidTy = int(midTip.x * w), int(midTip.y * h)
            cv2.circle(img, (MidTx, MidTy), 10, (100, 0, 255), cv2.FILLED)

            midDip = hands_landmarks.landmark[11]
            MidDx, MidDy = int(midDip.x * w), int(midDip.y * h)
            cv2.circle(img, (MidDx, MidDy), 10, (100, 0, 255), cv2.FILLED)

            midPip = hands_landmarks.landmark[10]
            MidPx, MidPy = int(midPip.x * w), int(midPip.y * h)
            cv2.circle(img, (MidPx, MidPy), 10, (100, 0, 255), cv2.FILLED)

            midMcp = hands_landmarks.landmark[9]
            MidMx, MidMy = int(midMcp.x * w), int(midMcp.y * h)
            cv2.circle(img, (MidMx, MidMy), 10, (100, 0, 255), cv2.FILLED)


            ThumbTip = hands_landmarks.landmark[4]
            ThmTx, ThmTy = int(ThumbTip.x * w), int(ThumbTip.y * h)
            cv2.circle(img, (ThmTx, ThmTy), 10, (100, 0, 255), cv2.FILLED)

            ThumbTip = hands_landmarks.landmark[4]
            ThmTx, ThmTy = int(ThumbTip.x * w), int(ThumbTip.y * h)
            cv2.circle(img, (ThmTx, ThmTy), 10, (100, 0, 255), cv2.FILLED)

            RingTip = hands_landmarks.landmark[16]
            RingTx, RingTy = int(RingTip.x * w), int(RingTip.y * h)
            cv2.circle(img, (RingTx, RingTy), 10, (100, 0, 255), cv2.FILLED)

            LittleTip = hands_landmarks.landmark[20]
            LteTx, LteTy = int(LittleTip.x * w), int(LittleTip.y * h)
            cv2.circle(img, (LteTx, LteTy), 10, (100, 0, 255), cv2.FILLED)

            # Conditions for Peace 
            # All Index finger and Middle finger y should be increasing downwards
            # Ring and Little finger Tip y must be greater than Middle finger Pip
            # Little Tip y is higher than or equal to Thumb Tip y
            # Thumb is lower than or equal to index finger mcp (y higher of thumb compared to index mcp y .. as its opposite) so that open thumb wont say its peace 

            if cy<dcy and dcy<pcy and pcy<mcy and MidTy<MidDy and MidDy<MidPy and MidPy<MidMy:
                if RingTy>MidPy and LteTy>MidPy and LteTy>=ThmTy and ThmTy >= mcy:
                    xOffset = w - overlay_size - 20
                    yOffset = 20
                    img[yOffset:yOffset+overlay_size,xOffset:xOffset+overlay_size] = peaceImage
                    # cv2.putText(img, "PEACE!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)





    cv2.imshow("MediaPipe Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import time


class handDetector():

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, 
                                        self.maxHands, 
                                        self.detectionCon, 
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        
        # Obtain hand values
        # print(results.multi_hand_landmarks)
        
        # Hand points
        if results.multi_hand_landmarks:
            # Draw points for every hand in the image
            for handLms in results.multi_hand_landmarks:

                if draw:
                    # HAND_CONNECTIONS: draw hand lines
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img
                # Hand poitns information
                # for id,lm in enumerate(handLms.landmark):

                #     # Decimals with respect picture dimensions
                #     # print(id,lm)

                #     h, w, c = img.shape
                #     cx, cy = int(lm.x*w), int(lm.y*h)
                #     print(id, cx, cy)

                #     # Show reference points (example 0: initial point, 4: last point of a finger)
                #     # if id == 4:
                #     cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)



def main():

    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    detector = handDetector()

    while True:
        success, img = cap.read()
        # Send our image to the method findHands()
        img = detector.findHands(img)

        # Show FPS
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    


if __name__ == '__main__':
    main()


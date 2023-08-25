import cv2
import mediapipe as mp



class HandLandmarks():
    def __init__(self,image_mode=False,maxhand=2,model_complexity=1,mindetecconf=0.5,mintrackconf=0.5):
        self.mode=image_mode
        self.hand=maxhand
        self.modelcomp=model_complexity
        self.deteconf=mindetecconf
        self.trackcon=mintrackconf

        self.mphands=mp.solutions.hands
        self.hands=self.mphands.Hands(self.mode,self.hand,self.modelcomp,self.deteconf,self.trackcon)
        self.draw=mp.solutions.drawing_utils

    def findhand(self,frame,draw=True):
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(framergb)
        if  self.result.multi_hand_landmarks:
            for Hand in  self.result.multi_hand_landmarks:
                if draw:
                    self.draw.draw_landmarks(frame, Hand, self.mphands.HAND_CONNECTIONS)
        return frame

    def handfindpos(self,frame,handno=0,draw=True):
        lmlist=[]
        if self.result.multi_hand_landmarks:
            myhands=self.result.multi_hand_landmarks[handno]
            for index, lmrk in enumerate(myhands.landmark):
                h, w, c = frame.shape
                lx, ly = int(lmrk.x * w), int(lmrk.y * h)
                # print(index,lx,ly)
                lmlist.append([index,lx,ly])
                #print(lmlist)
                if draw:
                    cv2.circle(frame, (lx,ly), 10, (255, 255, 255), -1)

        return lmlist
"""
def main():
    cam = cv2.VideoCapture(0)
    detec = HandLandmarks()
    while True:
        _,frame=cam.read()
        #print(result.multi_hand_landmarks)

        frame=detec.findhand(frame)
        lmlist=detec.handfindpos(frame,draw=False)
        if len(lmlist)!=0:
            pass
            #print(lmlist)


        cv2.imshow("camera",frame)
        if cv2.waitKey(1) & 0XFF==ord("q"):
            break

if __name__=="__main__":
    main()
"""

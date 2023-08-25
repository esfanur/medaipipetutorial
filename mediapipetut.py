import cv2
import mediapipe as mp
import mediapipe.python.solutions.drawing_utils

cam=cv2.VideoCapture(0)
mhands=mp.solutions.hands #"hands" modülüne erişim sağlamak ve el tespiti yapmak için bir nesne oluşturur.
hands=mhands.Hands() # mpHands nesnesi üzerinden el tespiti modelini başlatır.
mpdraw=mp.solutions.drawing_utils #MediaPipe kütüphanesinin içerdiği çizim işlevlerine erişim sağlamak ıcın
while True:
    _,frame=cam.read()

    framergb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=hands.process(framergb)
    #print(result.multi_hand_landmarks)
    if result.multi_hand_landmarks: #result degıskenıne atanmış tespit sonuçlarının el landmarks'larını içerip içermediğini kontrol eder. Eğer en az bir el tespit edilmişse, bu koşul doğru olur ve içeri giren blok çalıştırılır.
        """
        Bu kod parçası, el tespiti sonuçlarını alarak el landmarks'larını ve
        el arasındaki bağlantıları (connections) çizerek bir görüntü üzerine görselleştirmek için kullanılır
        """
        for hand in result.multi_hand_landmarks: #Eğer en az bir el tespit edilmişse, bu döngü tüm tespit edilen ellerin üzerinde döner
            for index,landmarks in enumerate(hand.landmark):#enumerate() fonksiyonu, hem dizin numarasını (index) hem de el landmarks'ının (landmarks) değerini döndürür.
                #print(index,landmarks)
                yuk,gen,k=frame.shape # Bu bilgileri  koordinat hesaplamalarında kullanacağız.yukseklıgı genıslıgı ve kanal sayısını alıyoruz
                kx,ky=int(landmarks.x*gen),int(landmarks.y*yuk) #x ve y koordinatlarını, görüntünün genişliği ve yüksekliği ile çarparak görüntü üzerindeki gerçek piksel koordinatlarına dönüştürüyoruz
                if index==4:
                    cv2.circle(frame,(kx,ky),10,(255,0,0),cv2.FILLED)
                    cv2.putText(frame, str(index), (kx, ky), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 4)

            mpdraw.draw_landmarks(frame,hand,mhands.HAND_CONNECTIONS) #frame, üzerine çizim yapılacak görüntüdür. hand, ilgili elin landmarks'larını içerir. mpHands.HAND_CONNECTIONS, el landmarks'ları arasındaki bağlantıları tanımlayan bir sabittir.

    cv2.imshow("video",frame)

    if cv2.waitKey(1) ==ord("q"):
        break
cam.release()
cv2.destroyAllWindows()
































"""
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # working with each hand
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 20:
                    cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
    cv2.imshow("Output", image)
    cv2.waitKey(1)
"""

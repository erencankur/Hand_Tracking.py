import cv2 # görüntü işleme görevlerini gerçekleştirmeye yarar
import mediapipe as mp # el takibi gibi çeşitli bilgisayar görüşü görevlerini gerçekleştirmeye yarar

camera = cv2.VideoCapture(0) # varsayılan kameraya erişmek için bir VideoCapture nesnesi oluşturuldu

mpHands = mp.solutions.hands # mediapipe kütüphanesinin hands modülünü mpHands olarak adlandırarak kullanmayı sağlar, el-parmak tespiti ve takibi yapmak için önceden eğitilmiş bir model sunar
hands = mpHands.Hands(False) # mpHands.Hands() sınıfı, el takibi işlemi için bir Hands nesnesi oluşturur.
mpDraw = mp.solutions.drawing_utils # mediapipe tarafından tespit edilen eklemler (landmarks) gibi yapıları görsel olarak çizmek için yardımcı işlevler ve çeşitli fonksiyonlar sunar

line_spec = mpDraw.DrawingSpec(color = (0, 255, 255), thickness = 2)
circle_spec = mpDraw.DrawingSpec(color = (255, 0, 0), thickness = -1, circle_radius = 8)

def count_fingers(hand_landmarks):
    finger_tips_numbers = [4, 8, 12, 16, 20]  # Baş parmak, işaret parmağı, orta parmak, yüzük parmağı, serçe parmak
    fingers_up = 0

    for tip_id in finger_tips_numbers:
        if tip_id != 4:
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                fingers_up += 1
        else:
            if hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x: # sol
                if hand_landmarks.landmark[4].x > hand_landmarks.landmark[2].x:
                    fingers_up += 1
            else: # sağ
                if hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x:
                    fingers_up += 1

    return fingers_up

while True: # sonsuz bir döngü oluşturularak kamera akışından sürekli olarak görüntü alınır 
    success, image = camera.read()  # camera.read() metodu kameradan bir kare (frame) yakalar, bu frame image'e atılır, succes ise True/False kontrolü yapar
    image = cv2.flip(image, 1) # simetrik olan görüntüyü çevirerek düzeltir
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV tarafından BGR formatında sağlanan görüntüyü RGB formatına dönüştürür
    results = hands.process(imageRGB) # process(imageRGB) metodu RGB formatındaki görüntüyü alıp el tespiti ve takibi işlemlerini gerçekleştirir, results değişkeni ise ellerin tespit edilip edilmediğini, kaç elin tespit edildiğini ve her bir eldeki parmak eklemlerinin 3D koordinatlarını içerir

    if results.multi_hand_landmarks: # her bir el için parmak eklemlerinin (landmarks) bilgilerini içeren bir liste döner
        total_fingers = 0

        for handLandmarks in results.multi_hand_landmarks: # bu döngü, tespit edilen her bir el için çalışır
            mpDraw.draw_landmarks(image, handLandmarks, mpHands.HAND_CONNECTIONS, circle_spec, line_spec) # tespit edilen eklemleri ve bağlantıları verilen görüntünün üzerine çizer
            finger_count = count_fingers(handLandmarks)
            total_fingers += finger_count
    else:
        total_fingers = 0
    
    cv2.putText(image, f"Total Fingers: {total_fingers}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5, cv2.LINE_AA)
    cv2.imshow("Image", image) # yakalanan görüntüyü Image adındaki bir pencerede gösterir
    cv2.waitKey(1) # her bir kare için bir milisaniyelik bekleme süresi koyar
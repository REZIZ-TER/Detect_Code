#สำรอง 1

from ultralytics import YOLO
import cv2
import cvzone
import math

_vdo = 'D:\Private\Y3Project\python_project\_assets\_alley.mp4'
cap = cv2.VideoCapture("rtsp://admin:kasidate01@169.254.25.21:554/Streaming/Channels/101/")

ptmodel = 'D:\Private\Y3Project\python_project\Weight3\model.pt'
model = YOLO(ptmodel)
className = ["Helmet", "Non_Helmet"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, t=3, colorR=(0, 255, 0), colorC=(0, 255, 0))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = className[cls]

            if currentClass == "Non_Helmet" and conf > 0.75:
                cvzone.putTextRect(img, f'{className[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=2.5,
                                   thickness=2, offset=8, colorR=(0, 0, 255), colorT=(255, 255, 255))

            if currentClass == "Helmet" and conf > 0.75:
                cvzone.putTextRect(img, f'{className[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=2.5,
                                   thickness=2, offset=8, colorR=(160, 255, 140), colorT=(0, 0, 0))

    cv2.imshow("Object Detection", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

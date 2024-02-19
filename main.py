from ultralytics import YOLO
import cv2
import cvzone

model = YOLO("Weight/best.pt")
nc = ['10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S', '4C', '4D', '4H', '4S', '5C', '5D',
      '5H', '5S', '6C', '6D', '6H', '6S', '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S', 'AC',
      'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS', 'QC', 'QD', 'QH', 'QS']
vid = cv2.VideoCapture(0)
vid.set(3, 640)
vid.set(4, 480)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("result/result_video.mp4",fourcc,30,(640,480))
while True:
    ret, frame = vid.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = int(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            class_name = nc[cls]

            cvzone.cornerRect(frame, [x1, y1, w, h],l=5)
            cvzone.putTextRect(frame, f"{class_name} {conf}", (x1, y1 - 10),offset=5,scale=1,thickness=2)
    out.write(frame)
    cv2.imshow("result", frame)
    if cv2.waitKey(1) == 27:
        break

out.release()
vid.release()
cv2.destroyAllWindows()

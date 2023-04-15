import cv2


pedestrian_cascade = cv2.CascadeClassifier('E:/WHJR PYTHON/PRO-106-ProjectTemplate-main/haarcascade_fullbody.xml')


cap = cv2.VideoCapture('E:/WHJR PYTHON/PRO-106-ProjectTemplate-main/walking.avi')


while True:
    
    ret, frame = cap.read()
    
    
    if not ret:
        break
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Pedestrian Detection', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

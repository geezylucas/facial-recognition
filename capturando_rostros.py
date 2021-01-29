import cv2
import os
import imutils

person_name = 'Gerardo'
data_path = '/Users/geezylucas/Documents/Python379.nosync/facial-recognition/data'
person_path = data_path + '/faces/' + person_name

if not os.path.exists(person_path):
    print('Carpeta creada: ', person_path)
    os.makedirs(person_path)

cap = cv2.VideoCapture(data_path + '/geezy.mov')

face_classif = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_frame = frame.copy()

    faces = face_classif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        rostro = aux_frame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(person_path + '/rostro_{}.jpg'.format(count), rostro)
        count = count + 1

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 300:
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import os
import numpy as np

data_path = '/Users/geezylucas/Documents/Python379.nosync/facial-recognition/data'
people_list = os.listdir(data_path + '/faces')

labels = []
faces_data = []
label = 0


for name_dir in people_list:
    person_path = data_path + '/faces/' + name_dir
    print('Leyendo imagenes')
    for file_name in os.listdir(person_path):
        print('Rostros: ', name_dir + '/' + file_name)
        labels.append(label)
        faces_data.append(cv2.imread(person_path + '/' + file_name, 0))
        image = cv2.imread(person_path + '/' + file_name, 0)

    label = label + 1


# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenado el reconocedor de rostros...
print('Entrenando....')

face_recognizer.train(faces_data, np.array(labels))

# Almacenando el modelo obtenido

# face_recognizer.write('modeloEigenFace.xml')
face_recognizer.write(data_path + '/modeloLBPHFace.xml')

print('Modelo almacenado...')

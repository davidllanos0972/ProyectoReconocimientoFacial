import cv2
import os 
import numpy as np 

dataPath = 'E:\semestre III\IOT Y ROBOTICA\ReconociminetoFacialPersona\Data'
peopleList = os.listdir(dataPath)
print('Lista de Personas: ',peopleList)

labels = []
facesData= []
label =0

for nameDir in peopleList:
    personPath = dataPath + '/'+nameDir
    print('Leyendo las imagenes')

    for fileName in os.listdir(personPath):
        print('Rostros: ',nameDir+'/'+fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
        image = cv2.imread(personPath+'/'+fileName, 0)
        #cv2.imshow('image', image)
        #cv2.waitKey(10)
    label = label +1
#print('Labels= ', labels)
#print('Numero de etiquetas 0: ', np.count_nonzero(np.array(labels)==0))
#print('Numero de etiquetas 1: ', np.count_nonzero(np.array(labels)==1))#

face_recognizer = cv2.face.EigenFaceRecognizer_create()

#Entrenando al reconocedor :v
print('Entrenando...')
face_recognizer.train(facesData, np.array(labels))

#Almacenando al modelo obtenido
face_recognizer.write('modeloEntrenado2.xml')
print('Modelo Almacendado')


cv2.destroyAllWindows()

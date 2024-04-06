import cv2, os
import numpy as np

dataPath = 'F:/FaceRecog/dataset/'

onlyFiles = [
    f for f in os.listdir(dataPath) if os.path.isfile(os.path.join(dataPath, f))
]

trainingData, labels = [], []

for i,files in enumerate(onlyFiles):
    imgPath= dataPath+onlyFiles[i]
    images=cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)

    trainingData.append(np.array(images,np.uint8))
    
    labels.append(i)

#print(trainingData)
labels=np.array(labels,dtype=np.int32)

model=cv2.face.LBPHFaceRecognizer_create()
model.train(trainingData,np.array(labels))


print("training successful")













"""
recognizer.save(dataPath+r'\trainer\trainer.yml')
cv2.destroyAllWindows()

PathList=os.listdir(dataPath)
imgList=[]
imgId=[]
for path in PathList:
    imgList.append(cv2.imread(os.path.join(dataPath,path)))
    imgId.append(os.path.splitext(path)[0])
"""

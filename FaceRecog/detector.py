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

    trainingData.append(np.array(images,'uint8'))
    
    labels.append(i)

labels=np.array(labels)

model=cv2.face.LBPHFaceRecognizer_create()
model.train(trainingData,np.array(labels))

print("trained!!!")
faceClassifier = cv2.CascadeClassifier("Face.xml")

#, size=0.5
def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(gray, 1.3, 5)

    if faces is():
        return img, []

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        croppedFace = img[y : y + h, x : x + w]
        croppedFace = cv2.resize(croppedFace, (200, 200))

    return img, croppedFace


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100 * (1 - (result[1]) / 300))
            cv2.putText(image,str(confidence)+"%", (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250,120,150), 2)

        if confidence > 80:
            cv2.putText(
                image,
                "arya",(250, 450),cv2.FONT_HERSHEY_COMPLEX,1,(255, 255, 255),2,
            )
            cv2.imshow("Face Cropper", image)

        else:
            cv2.putText(
                image,
                "Unknown",(250, 450),cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255),2,
            )
            cv2.imshow("Face Cropper", image)

    except:
        cv2.putText(
            image,
            "Face Not Found",(250, 450),cv2.FONT_HERSHEY_COMPLEX,1,(255, 0, 0),2,
        )
        cv2.imshow("Face Cropper", image)
        pass

    key=cv2.waitKey(1)    
    if (key==81 or key==113):
        break


cap.release()
cv2.destroyAllWindows()

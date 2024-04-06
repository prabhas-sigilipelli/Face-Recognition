import cv2,os

faceClassifier= cv2.CascadeClassifier('Face.xml') #loads dataset

def face_extractor(img):
    
    grayimg= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(grayimg, 1.3, 5)
    
    if faces is ():
        return None 
    
    for (x,y,w,h) in faces:
        croppedFace = img[y:y+h, x:x+w]
    return croppedFace

cap=cv2.VideoCapture(0)
count=0
name=input('enter name:')
while True:
    success,img=cap.read()
    if face_extractor(img) is not None:    
        count+=1
        face = cv2.resize(face_extractor(img), (200,200))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        
        filepath='F:/FaceRecog/dataset/'+name+'.'+str(count)+'.jpg'
        cv2.imwrite(filepath,face)
        cv2.imshow('Person',face)
    else:
        print("Face not found")
        pass

    key=cv2.waitKey(1)
    if(key==81 or key==113): #until Q key is pressed it runs
        break

cap.release()
cv2.destroyAllWindows()
print("Samples collected")
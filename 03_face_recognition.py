''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model (trained faces) should be on trainer/ dir


'''

import cv2



recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['none', 'Akash', 'id2', 'id 3', 'id 4', 'id 5'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 480
        ) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
c=input('\n Enter o to open')
while(True):
 
 if c=='o':
    print('in')
    ret, img =cam.read()
    img = cv2.flip(img, 1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        print("check")

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        conf=round(100 - confidence)
        # Check if confidence is less then 100 ==> "0" is perfect match 
        if ( conf > 30):
            id = names[id]
            confidence = "  {0}%".format(conf)
        else:
            id = "unknown"
            confidence = "  {0}%".format(conf)
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,0,0), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
        if id=='Akash':
                print('\n Access Granted')
                c='w'
        else:
                print('\n Access Denied')
                c='e'
        
    cv2.imshow('camera',img)
        
    k = cv2.waitKey(1) & 0xff # Press 'ESC' for exiting video
    if k==27:
        break
    
 elif(c=='w'):
            print('\n Welcome home')
 elif(c=='e'):
            print('\n Please Try Again')
            
        
    
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

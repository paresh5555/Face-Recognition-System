

# Import Statements
import cv2
import numpy as np

#Init Web Cam
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []
dataset_path = "./FRData/"
file_name = input('Enter the name of the person: ')

#Read the images
while True:
    ret, frame = cap.read()
    if ret==False:
        continue

    # convertng into grayscale
    gray_frame =  cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
     
    #situation where no faces detected
    if len(faces)==0:
        continue 

    #sorting the faces based on width * height = area and pick the last face as it has the largest area
    faces = sorted(faces,key=lambda f:f[2]*f[3])

    #picking from last
    for face in faces[-1:]:
        #draw bounding box or the rectangle
        x,y,w,h = face
        cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
        #showing the image
        cv2.imshow("Frame",frame)
        cv2.imshow("gray_frame",gray_frame)

        #extract or crop out the required face : region of interest
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        face_data.append(face_section)
        print(len(face_section))

    #our while loop will end when we press any key on the keyboard
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break

#convert face data list into numpy array 
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data)

#save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print('Data Saved Successfully')

#release the object and destroy all windows
cap.release()
cap.destroyAllWindows()




import cv2
import numpy as np
import os 
#Init Web Cam
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

###### KNN CODE ######
## distance formula between two numpy arrays - euclidean distance
def distance(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

#  KNN Algo
def knn(train,test,k=5):
# Pick k neraest neighbors
# for every point in the x 
    dist = []
    for i in range(train.shape[0]):
#         compute distance
        ix = train[i,:-1]
        iy = train[i,-1]
        d= distance(test,ix)
        dist.append((d,iy))
    
    
#     sort the array and find the k nearest points
    dk=sorted(dist,key=lambda x: x[0])[:k]
    #retrieve only the labels
    labels=np.array(dk)[:,-1]
    
#     Majority vote
    output = np.unique(labels,return_counts=True)
       
#     index of the maximum count
    index = np.argmax(output[1])
#     map this index with my data
    pred=output[0][index]
    return pred 


# Data Preparation
class_id = 0 # labels for the given file
names = {} # mapping id with name 
dataset_path = './FRData/'

face_data = []
labels = []

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
         #create a mapping between class id and name
         names[class_id] = fx[:4]
         data_item = np.load(dataset_path+fx)
         face_data.append(data_item)

         #create labels for the class
         target =  class_id*np.ones((data_item.shape[0],))

         class_id+=1
         labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels =  np.concatenate(labels,axis=0).reshape((-1,1))

train_set = np.concatenate((face_dataset,face_labels),axis=1)
print(train_set.shape)

#testing 
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

    #picking from last
    for face in faces[-1:]:
        #draw bounding box or the rectangle
        x,y,w,h = face
        
        #extract or crop out the required face : region of interest
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        
        # predict
        out=knn(train_set,face_section.flatten())

        # Display the output on the screen
        pred_names = names[int(out)]
        cv2.putText(gray_frame,pred_names,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(0,255,255),2)
       
    cv2.imshow("gray_frame",gray_frame)
    #our while loop will end when we press any key on the keyboard
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break

cap.release()
cap.destroyAllWindows()
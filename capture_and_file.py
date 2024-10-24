import PIL.Image
import PIL
import numpy as np
import cv2  #this is a module for life feed camera stuff 


"""
|| Contains all the image capturing functions and implamentation ||
!NOTE: ASSUMES EACH NAME HAS 5000 IMAGES UNDER THEM 
These image capture functions need a little bit of file set up before they work:
ideally you would have somethihng like this:
__________
testing
v training
    name1
    name2
    name3
__________

so then the image data can be put under each persons name
you can also have a different file set up, you just may have to make some changes to the functions
"""




"""
face_detector:
preloads in a face recongintion model to detect where a face is
"""
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


"""
capture_pictures_and_get_data:
takes n pictures and then returns the data from the cropped image in the form of a 2D array
(where each list is one images grayscale, down sized pixel data)
"""
def capture_pictures_and_get_data(n): #n is number of pictures to take
    final_image_data = []
    cam = cv2.VideoCapture(1)    #video capture object made
    cv2.namedWindow("Picture") #names the window 
    taking_pictures = True 
    while taking_pictures:
        ret, frame = cam.read()  #gets the camera 
        if not ret: 
            print("Failed to grab frame")
            break
        cv2.imshow("Picture", frame)  #shows the video/livefeed
        k = cv2.waitKey(1)  #tells it to wait for a key press
        if k%256 == 27:# ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:# SPACE pressed
            for x in range(n):    #for total numebr of images
                print(str(x) + " picture has been taken")
                picture = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #makes it grayscale while capturing what is displayed on the frame
                face_data = face_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)) #gets face data from the face detection model
                for (x, y, w, h) in face_data: #crops the picture from the face data
                    face = picture[y:y+h,x:x+w] 
                face = cv2.resize(face, (64, 64)) #this resizes the image to be of lower pixel count  to be the ratio you input
                image_cropped = PIL.Image.fromarray(face) #gets the frame data and gives it to a PIL image 
                values = list(image_cropped.getdata()) #gets the indidual pixel data as a list
                final_image_data.append(values)
            taking_pictures = False #finishs taking pictures
    cam.release()
    cv2.destroyAllWindows()
    return final_image_data #returns all the images pixel data
"""
write_info_to_file:
takes in a file path and some data in the form of a list
writes data to a file (split by whitespace)
"""
def write_info_to_file(file_path,data):
    file = open(file_path, "a") #opesn file 
    for x in data: 
        file.write(str(x)) #reads the data in
        file.write(" ") #space
    file.close() #closes the data
"""
write_all_info_to_file:
takes in the persons name and all the images related to that name along with what section of data it is (ie training or testing)
and then writes in all the images for that person
"""
def write_all_info_to_file(image_data,person_name,section):
    for x in range(len(image_data)): #for all the image data 
        file_name = str(person_name) + "_" + str(x) + ".txt" #makes a file name
        file_path = str(section) + "/" + person_name + "/" + file_name #makes the file path for this iteration 
        write_info_to_file(file_path,image_data[x]) #write to that file the image data for that one image

"""
retrive_file_info_by_person:
takes a persons name and returns all 5000 image data for that person and what section of the data this is, training or testing
!NOTE: ASSUMES EACH NAME HAS 5000 IMAGES UNDER THEM 
"""
def retrive_file_info_by_person(person_name,section):
    person_image_data = []
    for x in range(5000): #for all images #!NOTE: ASSUMES EACH NAME HAS 5000 IMAGES UNDER THEM 
        file = open(str(section) + "/" + str(person_name) + "/" + str(person_name) + "_" + str(x) + ".txt", "r") #open a file
        image_data = [] #temp image data
        for x in ((file.readline()).split()): #splits all the data in the file by whitespace (its all one long line in each file)
            image_data.append(int(x)) #adds each number indivudally
        person_image_data.append(image_data) #appends that data
        file.close()
    return person_image_data

"""
load_all_training_data:
takes in the list of peoples names and whether we are loading training or testing data and the output size of your model
returns the feature vectors and labels as two parrlel lists
#!NOTE: ASSUMES EACH NAME HAS 5000 IMAGES UNDER THEM 
"""
def load_all_data(list_of_people,section,output_size): 
    print("Loading all Data!")    
    all_data = [] #these two lists are in parallel
    all_label = []
    for x in range(len(list_of_people)):
        person_data = retrive_file_info_by_person(list_of_people[x],str(section)) #retireves all the training data for that person
        person_labels = np.zeros(output_size) #makes the label the same size as ouput size in model
        person_labels[x] = 1 #makes whatever person that is having there data loaded the label
        for y in range(5000): #!NOTE: ASSUMES EACH NAME HAS 5000 IMAGES UNDER THEM 
            all_data.append(person_data[y]) #gets the pixel data 
            all_label.append(person_labels) #adds the label 
    return all_data, all_label #returns all the data in two big lists 






n = 500 #!NUMBER OF PICTURES TO BE TAKEN

#write_all_info_to_file(capture_pictures_and_get_data(n),"other","training")

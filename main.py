import PIL.Image
import numpy as np
import PIL
import random
import numpy as np
import math
import capture_and_file
import additional_functions
import cv2  #this is a module for life feed camera stuff 
import time
list_of_people = ["bobby","noah"]

EPSILON = 0.00001 #small number to avoid divsion by zero
ALPHA = 0.01 #learning rate

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f'Time taken: {elapsed:.6f} seconds')
        return result
    return wrapper







class Model:
    layers = []
    epochs = 10
    def __init__(self, model_structure):
        self.structure = model_structure #this is the list of sizes of the various matrix's and 
        """
        the structure will be like this, [4096,3000,1500,750,350,100,5] 
        """
        for num_layers in range(len(model_structure)):
            if num_layers == 0:
                #layer = [np.random.randn(model_structure[num_layers],model_structure[num_layers+1])] #?this could be used
                layer = np.random.randn(model_structure[num_layers],model_structure[num_layers]) #this also isnt fully nessecary but it makes the first layer square
            else:
                layer = np.random.randn(model_structure[num_layers],model_structure[num_layers-1]) #the second sizing needs to match the previous layer so that it can accept a vector from it
            self.layers.append(layer)

    def find_loss_per_element(self,i,label,feature_vector):
        return label[i] - feature_vector[i] + EPSILON


    #feed forward the data
    def feed_forward(self,input_vec):
        temp_vec = input_vec
        for x in range(len(temp_vec)): #this flattens our values between 0 and 1 
            temp_vec[x] = float(temp_vec[x])/255.0
        for x in self.layers:
            temp_vec = np.dot(x,temp_vec)
            temp_vec = additional_functions.sigmoid_vec(temp_vec)
        return temp_vec
    
    #! ----YET TO BE FIXED----
    def back_propagate_and_train_no_batch(self,feature_vec, label_vec):
        temp_vec = feature_vec
        activations = np.array #this is a list of the activations of a particular layer
        activations.append(temp_vec)
        for x in range(len(temp_vec)): #this flattens our values between 0 and 1 
            temp_vec[x] = float(temp_vec[x])/255.0
        for x in self.layers:
            temp_vec = np.dot(x,temp_vec)
            activations.append(temp_vec)
            temp_vec = additional_functions.sigmoid_vec(temp_vec)
        prediciton = temp_vec
        #error for the output layer 
        DELTAS = []
        #!||-----||output layer deltas||-----||
        delta_output = []
        for x in range(self.layers[len(len(self.layers)-1)]): #for each entire in the last matrix in the list of layers ie the output matrix
            i = activations[len(activations)-1][x] #take the activations of the previous layer for each node      
            error = self.find_loss_per_element(x,label_vec,feature_vec) #calcuates error
            delta = error * additional_functions.sigmoid_derivative(i)
            delta_output.append(delta)
        DELTAS.append[delta_output]
        iterator_for_activations = 2 #?THIS MAY NEED TO BE 3, RECONSIDER LATER
        iterator_for_deltas = 0
        #!||-----||The rest of the deltas||-----||
        for x in range([len(self.layers - 2)]): #for each layer other then the last one
            delta_temp = []
            for y in range(len(self.layers[x])): #for each node in that layer
                i = activations[len(activations)-iterator_for_activations][y] #gets the activations for the layer before it of that node that we are considering 
                sigDir = additional_functions.sigmoid_derivative(i)
                weights_and_delta = np.dot(DELTAS[iterator_for_deltas],self.layers[x+1][:,y]) #gets the column vector for the layer ahead of it   #?x[:,0]
                delta_temp.append(sigDir * weights_and_delta) #add them to the list
            DELTAS.append(delta_temp) #adds that delta list to the greater delta list 
            iterator_for_activations += 1
            iterator_for_deltas +=1
        #!||-----||update weights||-----||
        iterator_for_big_deltas = 1
        for x in range(len(self.layers)): #for each layer (matrix)
            for y in range(len(self.layers[x])): #for each row in that matrix
                for z in range(len(self.layers[x][y])): #for each weight in that row
                    self.layers[x][y][z] += ALPHA * activations[x][z] * DELTAS[len(DELTAS)-iterator_for_big_deltas][y]
            iterator_for_big_deltas +=1
        #return DELTAS

    @timeit
    def back_propagate_and_return_changes(self,feature_vec, label_vec):
        print("Back Propagating and will return changes!")
        temp_vec = feature_vec
        for x in range(len(temp_vec)): #this flattens our values between 0 and 1 
            temp_vec[x] = float(temp_vec[x])/255.0
        activations = []  #this is a list of the activations of a particular layer
        #!||-----||feed forward||-----||
        activations.append(temp_vec)
        for x in self.layers:
            temp_vec = np.dot(x,temp_vec) 
            activations.append(temp_vec) 
            temp_vec = additional_functions.sigmoid_vec(temp_vec)  
        prediciton = temp_vec
        #error for the output layer 
        DELTAS = []
        #!||-----||output layer deltas||-----||
        print("Finding output deltas")
        delta_output = []
        for x in range(len(self.layers[(len(self.layers)-1)])): #for each entire in the last matrix in the list of layers ie the output matrix
            i = activations[len(activations)-1][x] #take the activations of the previous layer for each node       
            error = self.find_loss_per_element(x,label_vec,feature_vec) #calcuates error
            delta = error * additional_functions.sigmoid_derivative(i)
            delta_output.append(delta)
        DELTAS.append(delta_output)
        #!||-----||The rest of the deltas||-----||
        print("Finding the rest of the deltas")
        iterator_for_activations = 3
        iterator_for_deltas = 0
        for x in range((len(self.layers) - 2),-1,-1): #for each layer other then the last one starting at len(self.layers) - 2 and then going down till zero
            delta_temp = []
            for y in range(len(self.layers[x])): #for each node in that layer #? this is going the correct direction, 
                i = activations[len(activations)-iterator_for_activations][y] #gets the activation for the layer before it of that node that we are considering 
                sigDir = additional_functions.sigmoid_derivative(i)
                weights_and_delta = np.dot(DELTAS[iterator_for_deltas],self.layers[x+1][:,y]) #gets the column vector for the layer ahead of it   #x[:,0] 

                delta_temp.append(sigDir * weights_and_delta) #add them to the list 
            DELTAS.append(delta_temp) #adds that delta list to the greater delta list 
            iterator_for_activations += 1
            iterator_for_deltas +=1
        #!||-----||create desired changes to ouput for this image||-----||      BEING VERY SLOW THIS STEP SPECFICLLY
        print("creating desired changes")
        CHANGES = []
        for x in range(len(self.layers)): #for each layer (matrix)
            matrix = np.zeros(shape = (len(self.layers[x]),len(self.layers[x][0]))) #makes each CHANGES layer a matrix of zeros
            CHANGES.append(matrix)
            
        #?this is where we are gonna want to do the cplusplus code
        iterator_for_big_deltas = 1 
        for x in range(len(self.layers)): #for each layer (matrix)
            matrix = np.zeros(shape = (len(self.layers[x]),len(self.layers[x][0]))) #makes a matrix of zeros of the right size
            for y in range(len(self.layers[x])): #for each row in that matrix
                row = np.zeros(shape = len(self.layers[x][y]))
                for z in range(len(self.layers[x][y])): #for each weight in that row
                    row[z] = activations[x][z] * DELTAS[len(DELTAS)-iterator_for_big_deltas][y] #adds the desired changes to the row
                matrix[y] = row
            CHANGES[x] = (matrix)
            iterator_for_big_deltas +=1
        return CHANGES

    """ train:
    # takes in the list of images and list of labels for all the data along with the desired batch size
    # calcuates desired changes for each image(over a batch size), then sums these desired changes to weights to then ably to the weights
    # returns nothing, but will edit the models weights
    """
    def train(self,image_list,labels_list,batch_size): #number of epochs will be determined so that all the data is run through
        print("Beginning Training!")
        print("here is the number of images we have: " + str(len(image_list))) 
        num_epochs = len(image_list) / (batch_size) #number of batchs that will need to be mad
        batch_iterator = 0 #this iterator goes through all the images
        print("We are going to run through this many epochs: ")
        print(num_epochs)
        for x in range(int(num_epochs)): #for the number of epochs needed to get all the data 
            print("We are this many epochs in:" + str(x))
            batch_changes = [] #make a batch of desired changes to make to model weights
            for y in range(batch_size): #for each batch 
                changes = self.back_propagate_and_return_changes(image_list[batch_iterator],labels_list[batch_iterator]) #get desired changes for this image
                if y == 0: #if its the first, then its the starting changes 
                    batch_changes = changes 
                else:
                        #& changes is a big array that has the size of the weights list (ie layers)
                    print("adding elements along!")
                    batch_changes = additional_functions.add_arrays_by_element_3D(batch_changes,changes)
                batch_iterator += 1 
                print(str(batch_iterator) + " out of " + str(len(image_list)))
            for x in range(len(self.layers)): #for each layer (matrix)
                for y in range(len(self.layers[x])): #for each row in that matrix
                    for z in range(len(self.layers[x][y])): #for each weight in that row
                        self.layers[x][y][z] += ALPHA * batch_changes[x][y][z]  #add the desired changes for that
            
            






    def store_model(self,model_version_name):
        file = open((model_version_name + ".txt"), "a") #opesn file 
        for x in self.structure: #a
            file.write(str(x) + " ")
        file.write("\n")
        for x in self.layers: #for each layer
            for y in x:  #for each row in that matrix
                for z in y: #for each entire in that row
                    file.write(str(z)) #adds each weight entry 
                    file.write(" ")
                file.write("\n")#new line
        file.close() 

    def load_model(self,file_name):
        file = open(file_name, "r") #opesn file
        self.structure.clear()
        struc = (file.readline()).split() #gets the model structure
        for x in range(len(struc)):
            self.structure.append(int(struc[x]))  #adds the model structure to the model itself
        self.layers.clear()
        for x in self.structure: #for each layer
            matrix_temp = []
            for y in range(x): #for the number of rows in that layer
                row = (file.readline()).split()# gets the row data 
                for z in range(len(row)): #converts all to float
                    row[z] = float(row[z])
                matrix_temp.append(row) #adds the row
            self.layers.append(matrix_temp) #adds the matrix layer
        file.close()
            











    

#m1 = Model([5,4,3,2])
#m1.store_model("test1")
#m2 = Model([5,4])
#m2.load_model("test1.txt")



m1 = Model([4096,3000,1500,750,350,100,5])
all_images, all_labels = capture_and_file.load_all_data(list_of_people,"training",5) 
print(len(all_images))
m1.train(all_images,all_labels,100)
m1.store_model("1.3")
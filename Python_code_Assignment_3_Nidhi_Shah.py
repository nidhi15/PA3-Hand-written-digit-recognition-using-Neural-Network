#!/usr/bin/env python
# coding: utf-8

# In[387]:


# Northeastern University - Fall 2019 
# ALY6020 - Predictive Analytics
# Marco Montes de Oca

# Assignment 3 - Submitted By Nidhi Shah
print("********** Start of the Code ***********")


# In[388]:


#Import all the required libraries
import pandas as pd # data processing (e.g. pd.read_csv)
import numpy as np # linear algebra
import tensorflow as tf
import keras
import matplotlib.pyplot as plt  #Plotting the graph
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.models import Sequential    # Linear stack of layers
from keras.layers import Dense , Dropout  # Dense layers are "fully connected" layers
from keras.optimizers import Adam ,RMSprop  #Optimizers
from keras import  backend as K
from sklearn.metrics import confusion_matrix


# In[340]:


#Load the given train and test datasets 
train_data = pd.read_csv("mnist_train.csv", header = None)

test_data = pd.read_csv("mnist_test.csv", header = None)

#Glimpse of the data - first few rows of the actual data
train_data.head()
test_data.head()

#Dimensions of the datasets are shown using shape() function
print("Training Data shape is: ", train_data.shape)
print("Test Data shape is : ", test_data.shape)
print("Glimpse of the train data: \n", train_data.head(3), "\n")
print("Glimpse of the test data: \n", train_data.head(3))


# In[341]:


#Split the given train data to x_train and y_train 
x_train = train_data.loc[:,1:785]  #All 784 pixel values

#Reshaping the training pixels data to array  
x_trn = x_train.to_numpy()

print("Dimensions for training pixels are {}".format(x_trn.shape))
print("Glimpse of the first 5 rows: \n", x_trn[0:5], "\n")

y_train = train_data.loc[:,0] #1st column - Labels (digits)

#Reshaping the training labels data to array  
y_trn = y_train.to_numpy()

print("Dimensions for training labels are {}".format(y_trn[:,None].shape))
print("Glimpse of the first 5 labels:", y_trn[0:5])


# In[342]:


#Reshaping the given test dataset to array - split the given train data to x_test and y_test
x_test = test_data.loc[:,1:785]  #all pixel values
x_tst = x_test.to_numpy()
print("Dimensions for testing pixels are {}".format(x_tst.shape))
print("Glimpse of the first 5 rows: \n", x_tst[0:5], "\n")

y_test = test_data.loc[:,0]
y_tst = y_test.to_numpy()

print("Dimensions for testing labels are {}".format(y_tst[:,None].shape))
print("Glimpse of the first 5 labels:", y_tst[0:5])


# In[343]:


#Plot some of the Images
for i in range(10):
    img = x_trn[i].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()


# In[344]:


#Preparing the Data
# Normalize inputs from 0-255 to 0-1
x_trn = x_trn / 255
print("Normalized Train Pixels: \n", x_trn[0:5])
x_tst = x_tst / 255
print("Normlaized Test Pixels: \n", x_tst[:5])


# In[348]:


#Converting the labels(digits) in y_train and y_test to seven-segment display
#Numpy Zeros to represent the binary code for numbers between 0-9 - one-hot encode
d = np.zeros((10,7))
d[0] = [1,1,1,1,1,1,0]
d[1] = [0,1,1,0,0,0,0]
d[2] = [1,1,0,1,1,0,1]
d[3] = [1,1,1,1,0,0,1]
d[4] = [0,1,1,0,0,1,1]
d[5] = [1,0,1,1,0,1,1]
d[6] = [1,0,1,1,1,1,1]
d[7] = [1,1,1,0,0,0,0]
d[8] = [1,1,1,1,1,1,1]
d[9] = [1,1,1,1,0,1,1]
             
def digit_display(label):
    arr = np.zeros((len(label), 7))
    for i in range(len(label)):
        if label[i] in [0,1,2,3,4,5,6,7,8,9]:
            arr[i] = d[label[i]]
    return arr

#Apply the function to the y_trn and y_tst
y_dgt_trn = digit_display(y_trn)
print("The seven segment display for the digits in train dataset (y_dgt_trn) are:\n", y_dgt_trn, "\n")
y_dgt_tst = digit_display(y_tst)
print("The seven segment display for the digits in test dataset (y_dgt_tst) are:\n", y_dgt_tst)


# In[350]:


#Building the Model
image_size = 784 # 28*28
num_classes = 7 #7 segment display of the digits

model = Sequential()

#Hidden Layers
model.add(Dense(512, activation='relu', input_shape=(image_size,)))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
#Output Layer 
model.add(Dense(units=num_classes, activation='sigmoid'))

#Summary of the model
model.summary()

#Compiling the neural network
model.compile(optimizer = "adam" , loss = 'binary_crossentropy' , metrics = ['accuracy'] )

#Modeling on Training Dataset
history = model.fit(x_trn, y_dgt_trn, epochs = 4, batch_size = 128, validation_data=(x_tst,y_dgt_tst) )


# In[354]:


#Plot and Evaluate the model
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

#Plot grpah for Accuracy of the Model 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy of the Model')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='best')
plt.show()

#Plot graph for Loss of the Model
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss of the Model')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Training Loss', 'Validation Loss'], loc='best')
plt.show()

final_loss, final_acc = model.evaluate(x_tst, y_dgt_tst, verbose=0)
print("Final loss: {0:.3f}, Final accuracy: {1:.3f}".format(final_loss, final_acc))


# In[376]:


# Predict the values from the validation dataset
dgt_pred = model.predict([x_tst])
print(dgt_pred)
# Round-off to nearest 1
dgt_pred = round(pd.DataFrame(dgt_pred)) 
dgt_pred.head(3)


# In[380]:


#Converting the predicted values back to the labels(digits) for creating confusion matrix

#Numpy Zeros to represent the binary code for numbers between 0-9 - one-hot encode
d = np.zeros((10,7))
d[0] = [1,1,1,1,1,1,0]
d[1] = [0,1,1,0,0,0,0]
d[2] = [1,1,0,1,1,0,1]
d[3] = [1,1,1,1,0,0,1]
d[4] = [0,1,1,0,0,1,1]
d[5] = [1,0,1,1,0,1,1]
d[6] = [1,0,1,1,1,1,1]
d[7] = [1,1,1,0,0,0,0]
d[8] = [1,1,1,1,1,1,1]
d[9] = [1,1,1,1,0,1,1]

#Function for Remapping the one-hot encode back to numbers between 0-9 
def remap_label(label):
    a = pd.DataFrame(d)
    label = pd.DataFrame(label) 
    array = np.zeros((len(label), 1))
    for l in range(len(label)):
        for dgt in range(len(d)):
            if np.all(label.loc[l] == a.loc[dgt]):
                array[l] = dgt
    return array

#Pattern to Digit for predicted values of label and test data labels
y_pred_dgt = remap_label(dgt_pred)
y_tst_dgt = remap_label(y_dgt_tst)    
print("Predicted Labels \n", y_pred_dgt)
print("Test Labels \n", y_tst_dgt)
print("End of Modeling")


# In[383]:


#Confusion Matrix for the predicted values of label and test data labels
cm = confusion_matrix(y_tst_dgt,y_pred_dgt)

#Converting to a Data Frame
df_cm = pd.DataFrame(cm, range(len(cm)), range(len(cm)))
print("Confusion Matrix \n\n", df_cm)


# In[385]:


#Plot the Confusion Matrix
plt.matshow(df_cm, cmap=plt.cm.Blues) # imshow
plt.title("Confusion Matrix HeatMap \n \n")
plt.colorbar()
tick_marks = np.arange(len(df_cm.columns))
plt.xticks(tick_marks, df_cm.columns, rotation=50)
plt.yticks(tick_marks, df_cm.index)
#plt.tight_layout()
plt.ylabel("Actual")
plt.xlabel("Predicted")


# In[389]:


print("*********** End of the Code **************")


# In[ ]:





# PA3-Hand-written-digit-recognition-using-Neural-Network

Transform MNIST images into their corresponding seven segment display representation a feedforward neural network .

Two files are provided in the dataset. Each of these files contains thousands of 28x28 grayscale images as shown in the Image attached "Seven_segment_Image.png".Each image is encoded as a row of 784 integer values between 0 and 255 indicating the brightness of each pixel. The label associated with each image is encoded as an integer value between 0 and 9. The arrangement is shown in the picture "Arrangement_of_Data.png".

To train a feed forward neural network to classify each image into one of ten digits. The class the digit belongs to is not to be encoded by the activation of one out of ten output nodes. Instead, the idea is to represent the class as a pattern that could represent LEDs that when lit up at the same time in a seven segment display represent the digit itself. 

The network to train has therefore 784 inputs and 7 outputs. Try networks with one, two, or three hidden layers and with various number of neurons each. There is no fixed rules to decide how many layers, and how many neurons per layer are necessary to solve a problem. Also, the activation functions to use are very much determined by the nature of the problem to solve. In this case, sigmoid, or tanh neurons may be appropriate. Test the model on the test set. The output of the model will be the predicted digit given the image information. Create a confusion matrix  in order to capture the decisions made by the model.







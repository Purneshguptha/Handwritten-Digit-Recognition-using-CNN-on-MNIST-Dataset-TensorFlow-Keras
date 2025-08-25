#MNIST Classification
# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

#Load dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data()

#Normalize the pixel values
x_train = x_train/255.0

#Reshape for CNN input
x_train = x_train.reshape((-1,28,28,1))
x_test = x_test.reshape((-1,28,28,1))

#One-hot encode the labels
y_train_cat=to_categorical(y_train,10)
y_test_cat=to_categorical(y_test,10)

#Built the CNN model
model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])

#Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

#Train the model
model.fit(x_train,y_train_cat,epochs=5,validation_split=0.2,batch_size=64)

#Evaluate the model
test_loss, test_acc=model.evaluate(x_test,y_test_cat)
print(f"Test accuracy: {test_acc:.4f}")

#Predict on test data
predictions=model.predict(x_test)
predicted_labels=np.argmax(predictions,axis=1)

# Visualize predictions for first 10 test images
plt.figure(figsize=(12,6))
for i in range(10):
  plt.subplot(2,5,i+1)
  plt.imshow(x_test[i].reshape(28,28), cmap='gray')
  plt.title(f"Predicted: {predicted_labels[i]}")
  plt.axis('off')
plt.show()

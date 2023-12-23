#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


# Define constants
num_classes = 5  # Number of traffic object classes (e.g., car, pedestrian, bicycle, etc.)
image_size = (224, 224)  # Input image size (adjust as needed)
batch_size = 32


# In[ ]:


# Create a CNN model for object detection and classification
model = keras.Sequential([
    layers.Input(shape=(image_size[0], image_size[1], 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])


# In[ ]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# Create data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)


# In[ ]:


# Replace 'path/to/training_data' with the path to your training data directory
train_generator = train_datagen.flow_from_directory('path/to/training_data',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'path/to/training_data',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


# In[ ]:


# Train the model
epochs = 10  # Adjust as needed
model.fit(train_generator, validation_data=validation_generator, epochs=epochs)

# Save the trained model (optional)
model.save('traffic_object_detection_classification_model.h5')

# Use the trained model for traffic analysis
# You can use the model to predict the class of objects in traffic images


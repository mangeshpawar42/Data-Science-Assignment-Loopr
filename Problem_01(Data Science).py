#!/usr/bin/env python
# coding: utf-8

# # Implementation of Image Classification By using Computer Vision

# #### Importing Required library and Models

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.layers import BatchNormalization


# Dimension = 224*224

# In[26]:


Image_Size = 224
Channel = 3           # RGB
Batch_Size = 32
Epoch = 5


# #### Loading Dataset

# In[27]:


df =tf.keras.preprocessing.image_dataset_from_directory(
    "Data",
    image_size = (Image_Size,Image_Size),
    batch_size = Batch_Size,
    shuffle = True

)


# So, we have 651 images with 5 different classes

# In[28]:


# Let's check the classes

cls_name = df.class_names
cls_name


# In[29]:


len(df)


# Our 1 batch consist 32 image, so according to that length of dataset is 21

# In[30]:


for image_batch,label_batch in df.take(1):
    
    print(image_batch.shape)
    print(label_batch.numpy())


# ### Visualization

# In[31]:


plt.figure(figsize=(10,10))
for image_batch,label_batch in df.take(1):
    for i in range(15):
        ax = plt.subplot(5,3,i+1) 
        plt.imshow(image_batch[i].numpy().astype('uint8'))
        plt.axis('off')                                    # We can hide the axis by assigning '
        plt.title(cls_name[label_batch[i]])
    


# ### Training-Testing-Validation
# 
# Here, we split the data into train, val and test. 
# so, We split the data in below ratio.
# 
# Training --> 80%
# 
# Testing  --> 10%
# 
# Validation --> 10%
# 
# ##### Define the train_tets_split fucntion

# In[32]:


def train_test_split(df,train_split=0.8,test_split=0.1,val_split=0.1 ,shuffle=True,shuffle_size=1000):
    df_len = len(df)
    if shuffle:
        df = df.shuffle(shuffle_size,seed=15)
    train_size = int(train_split*df_len)
    val_size = int(val_split*df_len)
    train = df.take(train_size)
    val   = df.skip(train_size).take(val_size)
    test = df.skip(train_size).skip(val_size)
    return train,val,test


# In[33]:


train,val,test = train_test_split(df)


# In[34]:


train


# In[35]:


print("Length of Training set : ", len(train))
print("Length of Validation set : ", len(val))
print("Length of Test set : ", len(test))


# ##### For improving training performance, we are performing prefetch and cache. So, we can improve our model performance 

# In[36]:


train = train.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
val = val.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
test = test.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)


# ### Feature Scaling
# 
# - Performing Resizing and rescaling 

# In[37]:


resize_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(Image_Size,Image_Size),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])


# ### Data Augmentation 
# 
# - Improving training set by 4x
# - We can consider the all side view as Final Dataset

# In[38]:


# Data Augmentation

data_aug = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])


# ## Model Building
# 
# - Build 4 different model which are basically
# 
#    1. Normal CNN Architecture
#    2. ResNet50
#    3. ResNet101
#    4. AlexNet

# ### Normal Architecture

# In[39]:


# Normal CNN Model

n_classes = 5
Input = (Batch_Size,Image_Size,Image_Size,Channel)

model = models.Sequential([
    resize_rescale,
    data_aug,
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(Image_Size,Image_Size)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(n_classes,activation='softmax')
])

model.build(input_shape=Input)
model.summary()


# In[40]:


model.compile(
    optimizer='adam',
    loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics = ['accuracy']
)


# In[41]:


# Normal Model
Normal_history = model.fit(
    train,
    epochs = Epoch,
    batch_size = Batch_Size,
    verbose = 1,
    validation_data = val
    )


# In[45]:


model_score = model.evaluate(test)
model_score
print(f"Normal CNN Test Score: {np.round(model_score[1]*100,2)}%")


# ### AlexNet

# In[51]:


# AlexNet

model_alex=Sequential()

#1 conv layer
model_alex.add(Conv2D(filters=96,kernel_size=(11,11),strides=(4,4),padding="valid",activation="relu",input_shape=(224,224,3)))

#1 max pool layer
model_alex.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model_alex.add(BatchNormalization())

#2 conv layer
model_alex.add(Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),padding="valid",activation="relu"))

#2 max pool layer
model_alex.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model_alex.add(BatchNormalization())

#3 conv layer
model_alex.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding="valid",activation="relu"))

#4 conv layer
model_alex.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding="valid",activation="relu"))

#5 conv layer
model_alex.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding="valid",activation="relu"))

#3 max pool layer
model_alex.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model_alex.add(BatchNormalization())


model_alex.add(Flatten())

#1 dense layer
model_alex.add(Dense(4096,input_shape=(227,227,3),activation="relu"))

model_alex.add(Dropout(0.4))

model_alex.add(BatchNormalization())

#2 dense layer
model_alex.add(Dense(4096,activation="relu"))

model_alex.add(Dropout(0.4))

model_alex.add(BatchNormalization())

#3 dense layer
model_alex.add(Dense(1000,activation="relu"))

model_alex.add(Dropout(0.4))

model_alex.add(BatchNormalization())

#output layer
model_alex.add(Dense(20,activation="softmax"))

model_alex.summary()



model_alex.compile(
    optimizer='adam',
    loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics = ['accuracy']
)

print(model_alex.summary())


# In[52]:


AlexNet_history = model_alex.fit(train,
                                 validation_data=val,
                                 epochs=Epoch,
                                 batch_size=Batch_Size,
                                 verbose=1)



# In[53]:


model_score_alex = model_alex.evaluate(test)
model_score_alex
print(f"AlexNet Test Accuracy : {np.round(model_score_alex[1]*100,2)}%")


# In[ ]:





# ### ResNet50

# In[46]:


# ResNet50

resnet = tf.keras.applications.resnet50.ResNet50(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet',
                    pooling='avg')

resnet.trainable = False

x = tf.keras.layers.Dense(128, activation='relu')(resnet.output)
x = tf.keras.layers.Dense(50, activation='relu')(x)
outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
model_resnet = tf.keras.Model(resnet.input, outputs)

model_resnet.compile(
    optimizer='adam',
    loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics = ['accuracy']
)

print(model_resnet.summary())


# In[47]:


ResNet_history = model_resnet.fit(train,
                                  validation_data=val,
                                  epochs=Epoch,
                                  batch_size=Batch_Size,
                                  verbose=1)


# In[48]:


# ResNet 50
ResNet_result = model_resnet.evaluate(test, verbose=0)
print(ResNet_result)
print(f"ResNet50 Test Accuracy: {np.round(ResNet_result[1] * 100,2)}%")


# In[49]:


# ResNet101

resnet101 = tf.keras.applications.ResNet101(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet',
                    pooling='avg')

resnet101.trainable = False

x = tf.keras.layers.Dense(128, activation='relu')(resnet101.output)
x = tf.keras.layers.Dense(50, activation='relu')(x)
outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
model_resnet101 = tf.keras.Model(resnet101.input, outputs)

model_resnet101.compile(
    optimizer='adam',
    loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics = ['accuracy']
)

print(model_resnet101.summary())

ResNet101_history = model_resnet101.fit(train,
                                  validation_data=val,
                                  epochs=Epoch,
                                  batch_size=Batch_Size,
                                  verbose=1)


# In[50]:


# ResNet 101
ResNet101_result = model_resnet101.evaluate(test, verbose=0)
print(ResNet101_result)
print(f"ResNet101 Test Accuracy: {np.round(ResNet101_result[1] * 100,2)}%")


# In[54]:


Normal_history.params


# # Plotting the graph
# 
# - Accuracy vs val_accuracy
# - Loss Vs Val_Accuracy

# In[57]:


accuracy_Normal = Normal_history.history['accuracy']
val_accuracy_Normal = Normal_history.history['val_accuracy']

loss_Normal = Normal_history.history['loss']
val_loss_Normal = Normal_history.history['val_loss']


# In[58]:


plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(range(Epoch),accuracy_Normal,label='Normal_Model_Training_Accuracy')
plt.plot(range(Epoch),val_accuracy_Normal,label='Normal_Model_Validation_Accuracy')
plt.legend()
plt.title("Training and Validation Accuracy For ResNet101")

plt.subplot(1,2,2)
plt.plot(range(Epoch),loss_Normal,label='TNormal_Model_raining_Loss')
plt.plot(range(Epoch),val_loss_Normal,label='Normal_Model_Validation_Loss')
plt.legend()
plt.title("Training and Validation Loss For Normal_Model")
plt.show()


# In[59]:


accuracy_Res101 = ResNet101_history.history['accuracy']
val_accuracy_Res101 = ResNet101_history.history['val_accuracy']

loss_Res101 = ResNet101_history.history['loss']
val_loss_Res101 = ResNet101_history.history['val_loss']


# In[60]:


plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(range(Epoch),accuracy_Res101,label='ResNet101_Training_Accuracy')
plt.plot(range(Epoch),val_accuracy_Res101,label='ResNet101_Validation_Accuracy')
plt.legend()
plt.title("Training and Validation Accuracy For ResNet101")

plt.subplot(1,2,2)
plt.plot(range(Epoch),loss_Res101,label='Training_Loss')
plt.plot(range(Epoch),val_loss_Res101,label='Validation_Loss')
plt.legend()
plt.title("Training and Validation Loss For ResNet101")
plt.show()


# In[61]:


accuracy_Res50 = ResNet_history.history['accuracy']
val_accuracy_Res50 = ResNet_history.history['val_accuracy']

loss_Res50 = ResNet_history.history['loss']
val_loss_Res50 = ResNet_history.history['val_loss']

# Ploting Graph

plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(range(Epoch),accuracy_Res101,label='ResNet50_Training_Accuracy')
plt.plot(range(Epoch),val_accuracy_Res101,label='ResNet50_Validation_Accuracy')
plt.legend()
plt.title("Training and Validation Accuracy For ResNet50")

plt.subplot(1,2,2)
plt.plot(range(Epoch),loss_Res101,label='ResNet50_Training_Loss')
plt.plot(range(Epoch),val_loss_Res101,label='ResNet50_Validation_Loss')
plt.legend()
plt.title("Training and Validation Loss For ResNet50")
plt.show()


# In[62]:


accuracy_Alex = AlexNet_history.history['accuracy']
val_accuracy_Alex = AlexNet_history.history['val_accuracy']

loss_Alex =AlexNet_history.history['loss']
val_loss_Alex = AlexNet_history.history['val_loss']

# Ploting Graph

plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(range(Epoch),accuracy_Alex,label='AlexNet_Training_Accuracy')
plt.plot(range(Epoch),val_accuracy_Alex,label='AlexNet_Validation_Accuracy')
plt.legend()
plt.title("Training and Validation Accuracy For AlexNet")

plt.subplot(1,2,2)
plt.plot(range(Epoch),loss_Alex,label='AlexNet_Training_Loss')
plt.plot(range(Epoch),val_loss_Alex,label='AlexNet_Validation_Loss')
plt.legend()
plt.title("Training and Validation Loss For AlexNet")
plt.show()


# ### Test Case Prediction By using Best Accuracy model - ResNet101

# In[63]:


for image_batch, label_batch in test.take(1):
    first_image = image_batch[0].numpy().astype('uint8')
    first_label = label_batch[0].numpy()
    
    print('Test Cases For Prediction')
    print("Actual_Image : ",cls_name[first_label])
    plt.imshow(first_image)
    
    prediction = model_resnet101.predict(image_batch)
    print('Pridicted_Image : ', cls_name[np.argmax(prediction[0])])
    


# In[64]:


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model_resnet101.predict(img_array)

    predicted_class = cls_name[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


# In[65]:


plt.figure(figsize=(12,15))
for images, labels in test.take(1):
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, confidence = predict(model_resnet101, images[i].numpy())
        actual_class = cls_name[labels[i]] 
        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        
        plt.axis("off")


# # Saving Model
# 

# In[66]:


model.save('Normal_Architecture.h5')


# In[67]:


model_resnet.save("ResNet_Architecture.h5")


# In[68]:


model_resnet101.save('ResNet101_Architecture.h5')


# In[71]:


model_alex.save("AlexNet_Architecture.h5")


# ## Conclusion : 
# 
# Finally, I conclude that 
#   - I used 4 different CNN architecture for improving models accuracy
#   - Initially, I used large number of Epoch, for that my model give me the best accuracy but it take the time. 
#   - So, I decrease the Epoch. It's fine that my model's accuracy decrease. But it's not consuming the lots of time.
#   - Resnet50 outperforming with respect to other models.
#   
#   - Normal Architecture - 20.83%
#   - ResNet50 Architecture - 92.71%
#   - ResNet101 Architecture - 91.67%
#   - AlexNet Architecture - 25%

# In[ ]:





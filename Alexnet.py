from tensorflow.keras.layers import Input, Conv2D, \
     BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
import keras
import keras.backend as K

train_data_dir = 'data/train'
validation_data_dir = 'data/test'
nb_train_samples = 140
nb_validation_samples = 60
epochs = 40
batch_size = 2
#Compare epoch and batch_size (can be chaged)

img_width, img_height = 224, 224
input = Input(shape=(img_width, img_height, 3))

x = Conv2D(filters=96,
           kernel_size=11,
           strides=4,
           padding='same',
           activation='relu')(input)  # 1st convolutional layer
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=3, strides=2)(x)

x = Conv2D(filters=256,
           kernel_size=5,
           padding='same',
           activation='relu')(x)  # 2nd convolutional layer
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=3, strides=2)(x)

x = Conv2D(filters=384,
           kernel_size=3,
           padding='same',
           activation='relu')(x)  # 3rd convolutional layer

x = Conv2D(filters=384,
           kernel_size=3,
           padding='same',
           activation='relu')(x)  # 4th convolutional layer

x = Conv2D(filters=256,
           kernel_size=3,
           padding='same',
           activation='relu')(x)  # 5th convolutional layer
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=3, strides=2)(x)

x = Flatten()(x)
x = Dense(units=4096, activation='relu')(x)
x = Dense(units=4096, activation='relu')(x)
x = Dropout(rate=0.5)(x)

#ubah unit folder - 2 class kalau ada 2
output = Dense(units=2, activation='softmax')(x)
model = Model(inputs=input, outputs=output)

sgd = optimizers.SGD()
model.compile(loss ='categorical_crossentropy', 
                     optimizer = sgd, 
                   metrics =['accuracy'])
model.summary()
train_datagen = ImageDataGenerator( 
                rescale = 1. / 255)
  
test_datagen = ImageDataGenerator(rescale = 1. / 255) 

train_generator = train_datagen.flow_from_directory(train_data_dir, 
                              target_size =(img_width, img_height), 
                     batch_size = batch_size, class_mode ='categorical') 
  
validation_generator = test_datagen.flow_from_directory( 
                                    validation_data_dir, 
                   target_size =(img_width, img_height), 
          batch_size = batch_size, class_mode ='categorical') 

model.fit(train_generator, 
    steps_per_epoch = nb_train_samples // batch_size, 
    epochs = epochs, validation_data = validation_generator, 
    validation_steps = nb_validation_samples // batch_size)

# Save the model
model.save('Alexnet_model.h5')

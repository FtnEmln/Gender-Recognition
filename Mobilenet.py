from tensorflow.keras.layers import Input, DepthwiseConv2D, \
     Conv2D, BatchNormalization, ReLU, AvgPool2D, Flatten, Dense
from tensorflow.keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
import keras
import keras.backend as K

train_data_dir = 'data/train'
validation_data_dir = 'data/test'
nb_train_samples = 140
nb_validation_samples = 60
epochs = 100
batch_size = 2
#Change epoch and batch size
img_width, img_height = 224, 224
def mobilenet_block(x, filters, strides):
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

input = Input(shape=(img_width, img_height, 3))
x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(input)
x = BatchNormalization()(x)
x = ReLU()(x)

x = mobilenet_block(x, filters=64, strides=1)

x = mobilenet_block(x, filters=128, strides=2)
x = mobilenet_block(x, filters=128, strides=1)

x = mobilenet_block(x, filters=256, strides=2)
x = mobilenet_block(x, filters=256, strides=1)

x = mobilenet_block(x, filters=512, strides=2)
for _ in range(5):
    x = mobilenet_block(x, filters=512, strides=1)
  
x = mobilenet_block(x, filters=1024, strides=2)
x = mobilenet_block(x, filters=1024, strides=1)

x = AvgPool2D(pool_size=7, strides=1)(x)
x = Flatten()(x)
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
model.save('Mobilenet_model.h5')

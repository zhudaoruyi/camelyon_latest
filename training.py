from generator import get_picture, effective_list, data_generator
from keras.models import Sequential, load_model
from keras.layers import Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import CSVLogger, ModelCheckpoint
import os

epochs = 1000

if os.path.isfile('model_0823.h5'):
    model = load_model('model_0823.h5')

if os.path.isfile('training_0823.log'):
    log = open('training_0823.log')
    line = log.readlines()
    initial_epoch = (list(eval(line[-1])))[0]+1
else:
    initial_epoch = 0

# 训练集、验证集、测试集严密隔离
train_set = get_picture("train/tumor/origin_images/")    # 训练集数据文件夹
valid_set = get_picture("validation/")    # 验证集数据文件夹
# test_set = get_picture("test/")    # 测试集数据文件夹
mask_pictures = get_picture("train/tumor/annotation_images/")    # 所有的mask图文件夹


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(256, 256, 3)))    # 输入处理
model.add(Convolution2D(100, (3, 3), strides=(2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Convolution2D(200, (3, 3), strides=(2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Convolution2D(300, (3, 3), activation='relu', padding='same'))
model.add(Convolution2D(300, (3, 3), activation='relu', padding='same'))

model.add(Dropout(0.1))
model.add(Convolution2D(2, (1, 1))) # this is called upscore layer for some reason?
model.add(Conv2DTranspose(2, (31, 31), strides=(16, 16), activation='softmax', padding='same'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

csvlogger = CSVLogger('training_0823.log', append=True)
model_check = ModelCheckpoint('model_0823.h5')

model.fit_generator(data_generator(train_set),
                    steps_per_epoch=100,
                    epochs=epochs,
                    validation_data=data_generator(valid_set),
                    validation_steps=20,
                    verbose=1,
                    workers=10,
                    max_q_size=1,
                    callbacks=[csvlogger, model_check],
                    initial_epoch=initial_epoch)

model.save('model_0823.h5')

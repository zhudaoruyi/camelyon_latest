from generator import get_picture, effective_list, data_generator
from keras.models import Model, load_model
from keras.layers import Dense, Lambda, Dropout, GlobalAveragePooling2D
from keras.callbacks import CSVLogger, ModelCheckpoint
import os
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.optimizers import SGD


epochs = 100
widths, heights = 256,256
FC_SIZE = 1024
classes = 2

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

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(widths, heights, 3))    # 迁移学习，载入InceptionV3的权重，拿来直接用

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(FC_SIZE, activation='relu')(x)  # new FC layer, random init
predictions = Dense(classes, activation='softmax')(x)  # new softmax layer
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer=SGD(lr=0.05, momentum=0.9, decay=0.5), loss='categorical_crossentropy', metrics=['accuracy'])

csvlogger = CSVLogger('training_0823.log', append=True)
model_check = ModelCheckpoint('model_0823.h5')

model.fit_generator(data_generator(train_set, batch=8),
                    steps_per_epoch=100,
                    epochs=epochs,
                    validation_data=data_generator(valid_set, batch=8),
                    validation_steps=20,
                    verbose=1,
                    workers=10,
                    max_q_size=1,
                    callbacks=[csvlogger, model_check],
                    initial_epoch=initial_epoch)

model.save('model_0823.h5')

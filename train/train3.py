'''
    训练三：no weights and use SGD
'''

from generator2 import *
from keras.models import Model, load_model
from keras.layers import Dense, Lambda, Dropout, GlobalAveragePooling2D
from keras.callbacks import CSVLogger, ModelCheckpoint
import os
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.optimizers import SGD, RMSprop

epochs = 500
batch_size = 64
widths, heights = 299, 299

# 训练集、验证集、测试集严密隔离
train_set = get_picture("train/tumor/origin_images/")    # 训练集数据文件夹
valid_set = get_picture("validation/")    # 验证集数据文件夹
mask_pictures = get_picture("train/tumor/annotation_images/")    # 所有的mask图文件夹

base_model = InceptionV3(weights=None, include_top=False, input_shape=(widths, heights, 3), classes=2)    # 迁移学习，载入InceptionV3的权重，拿来直接用

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # new FC layer, random init
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)  # new softmax layer
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = True
model.compile(optimizer=SGD(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#model = load_model('model_1000_8_13.h5')

#model.compile(optimizer=SGD(lr=0.00007, momentum=0.5, decay=0.05), loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=RMSprop(lr=0.05,epsilon=1.0,decay=0.5), loss='categorical_crossentropy', metrics=['accuracy'])

csvlogger = CSVLogger('log50064100_03.log', append=True)
model_check = ModelCheckpoint('model50064100_03_p.h5', monitor='val_loss', save_best_only=True)

model.fit_generator(data_generator(train_set, batch=batch_size, widths=widths, heights=heights),
                    steps_per_epoch=100,
                    epochs=epochs,
                    validation_data=data_generator(valid_set, batch=batch_size, widths=widths, heights=heights),
                    validation_steps=100,
                    verbose=1,
                    workers=100,
                    max_q_size=64,
                    callbacks=[csvlogger, model_check])

model.save('model50064100_03.h5')

from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras.models import load_model, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications import ResNet50, VGG16, InceptionV3
from keras.applications.vgg16 import preprocess_input, decode_predictions

import os  #APIs
import sys
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
from keras.utils import np_utils#to_categorical
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#load dataset
train_size = 60000#20
val_size = 5000#10
test_size = 5000#5
data_test = np.zeros((test_size*2, 224,224,3),dtype=np.uint8)
data_train = np.zeros((train_size*2, 224,224,3),dtype=np.uint8)#np.zeros((120000, 224,224,3),dtype=np.uint8)
data_val = np.zeros((val_size*2, 224,224,3),dtype=np.uint8)#np.zeros((10000, 224,224,3),dtype=np.uint8)
label_train = np.array([0]*train_size +[1]*train_size)
label_val = np.array([0]*val_size +[1]*val_size)
label_test = np.array([0]*test_size+[1]*test_size)
label_test_predict = []

path_train_real = '/media/hung/YYQ/CAM_VGG16/dataset/faces/ffhq/real-1024-resized-224/train'
path_train_fake = '/media/hung/YYQ/CAM_VGG16/dataset/faces/ffhq/sgan-1024-resized-224/train'
path_val_real = '/media/hung/YYQ/CAM_VGG16/dataset/faces/ffhq/real-1024-resized-224/val'
path_val_fake = '/media/hung/YYQ/CAM_VGG16/dataset/faces/ffhq/sgan-1024-resized-224/val'
path_test_real = '/media/hung/YYQ/CAM_VGG16/dataset/faces/ffhq/real-1024-resized-224/test'
path_test_fake = '/media/hung/YYQ/CAM_VGG16/dataset/faces/ffhq/sgan-1024-resized-224/test'
results_output_path = '/media/hung/YYQ/CAM_VGG16/results2'
#namelist
train_real_list = os.listdir(path_train_real)
train_fake_list = os.listdir(path_train_fake)
val_real_list = os.listdir(path_val_real)
val_fake_list = os.listdir(path_val_fake)
test_real_list = os.listdir(path_test_real)
test_fake_list = os.listdir(path_test_fake)

#open images and img_to_array, then store it
#real_train
for i, filename in tqdm(enumerate(train_real_list), total=train_size):
    img = Image.open(path_train_real+'/'+filename)
    img = img_to_array(img, dtype=np.uint8)
    data_train[i] = img
#fake_train
for i, filename in tqdm(enumerate(train_fake_list), total=train_size):
    img = Image.open(path_train_fake+'/'+filename)
    img = img_to_array(img, dtype=np.uint8)
    data_train[i+train_size] = img
#real_val
for i, filename in tqdm(enumerate(val_real_list), total=val_size):
    img = Image.open(path_val_real+'/'+filename)
    img = img_to_array(img, dtype=np.uint8)
    data_val[i] = img
#fake_val
for i, filename in tqdm(enumerate(val_fake_list),total=val_size):
    img = Image.open(path_val_fake+'/'+filename)
    img = img_to_array(img, dtype=np.uint8)
    data_val[i+val_size] = img
#real_test
for i, filename in tqdm(enumerate(test_real_list), total=test_size):
    img = Image.open(path_test_real+'/'+filename)
    img = img_to_array(img, dtype=np.uint8)
    cv2.putText(img, 'gt: real', (150,18), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    data_test[i] = img
#fake_test
for i, filename in tqdm(enumerate(test_fake_list), total=test_size):
    img = Image.open(path_test_fake+'/'+filename)
    img = img_to_array(img, dtype=np.uint8)
    cv2.putText(img, 'gt: fake', (150,18), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    data_test[i+test_size] = img
#shuffle
#train
np.random.seed(116)
np.random.shuffle(data_train)
np.random.seed(116)
np.random.shuffle(label_train)
#val
np.random.seed(42)
np.random.shuffle(data_val)
np.random.seed(42)
np.random.shuffle(label_val)
#test
np.random.seed(42)
np.random.shuffle(data_test)
np.random.seed(42)
np.random.shuffle(label_test)
label_test = label_test.tolist()
#add
#data = np.concatenate((data_train, data_test), axis = 0)

#or shuffle by
#index = [i for i in range(len(data))]
#np.random.shuffle(index)
#data = data[index]
#label = label[index]

#create model (conv in VGG16 + GAP + dropout + Dense)
base_model = VGG16(include_top = False, weights = 'imagenet')
for layer in base_model.layers:
    layer.trainable = False #transfer learning (512+1)
GAP_layer = GlobalAveragePooling2D()(base_model.output)
drop = Dropout(0.25)(GAP_layer)
#flatten = Flatten()(drop)
y = Dense(2, activation='softmax')(drop)
model = Model(inputs = base_model.input, outputs = y)
model.summary()
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(x = data_train, y = label_train, validation_data=(data_val, label_val), batch_size=5, epochs=1)

# train model (conv can not be trained, so just depend on 513 weights)

label_train = np_utils.to_categorical(label_train,num_classes=2)#one-hot
label_val = np_utils.to_categorical(label_val, num_classes=2)
model.fit(x = data_train, y = label_train, batch_size=32, epochs=1, validation_data=(data_val, label_val))
weight_softmax = model.layers[-1].get_weights()[0]

#weights
#input images ---> feature maps(after vgg16 conv layers) ---> feature vector(after GAP) ---> classification rate(after weight matrix and sigmoid)
#for a picture, because base model not change which lead to Conv-output fixed (out_base)

target = data_test[1][:,:,::-1]
out_base = base_model.predict(np.expand_dims(target, axis = 0))
out_base = out_base[0]
print(out_base.shape)
#(7,7,512)

def predict_on_weights(out_base, weights):
    gap = np.average(out_base, axis=(0,1))
    logit = np.dot(gap, np.squeeze(weights))
    return 1 / (1 + np.e ** (-logit))


def save_img(img, file_name, file_path):
    try:
        if not os.path.exists(file_path):
            print('files', file_path, 'not exist and establish')
            os.makedirs(results_output_path, exist_ok=True)
        #file_suffix = os.path.splitext(img)[1]
        #file_suffix = 'png'
        #filename = '{}{}{}{}'.format(file_path, os.sep, file_name, file_suffix)
        cv2.imwrite(os.path.join(file_path, '%09d.%s' % (file_name, 'png')), img)
    except IOError as e:
        print('operation file error', e)
    except Exception as e:
        print('error:', e)

def image_compose(imgs):
    #input is a list of same size images
    h, w, _ = imgs[0].shape
    #create new image, and horizental compose
    result = np.zeros((h,w*2,_), dtype=np.uint8)
    for i, im in enumerate(imgs):
        result[0:h,w*i:w*(i+1)] = im
    return result

import seaborn as sns
band = np.array([list(np.arange(0,255,10))] * 1)
sns.heatmap(band, annot=True, fmt='d', cmap='jet', cbar=False)

#CAM computation

def getCAM(img, feature_maps, weights, class_idx, display = False):
    output_cam = []
    #accuracy = []
    nc, h, w = feature_maps.shape
    for idx in range(class_idx):
        weight = weights[:,idx].reshape(w,1)
        predict = predict_on_weights(feature_maps, weight)
        # weighted feature map
        cam = np.matmul(feature_maps, weight)
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        # Resize as image size
        cam_resize = cv2.resize(cam, (224, 224))  # this is same as input
        # format as CV_8UC1
        cam_resize = 255 * cam_resize
        cam_resize = cam_resize.astype(np.uint8)
        # Get heatmap
        heatmap = cv2.applyColorMap(cam_resize, cv2.COLORMAP_JET)
        # zero_out
        heatmap[np.where(cam_resize <= 100)] = 0

        out = cv2.addWeighted(src1=img, alpha=0.8, src2=heatmap, beta=0.4, gamma=0)

        if predict < 0.5:
            text = 'real %.2f%%' % (100 - predict * 100)
            #label_test_predict.append(0)
        else:
            text = 'fake %.2f%%' % (predict * 100)
            #label_test_predict.append(1)
        cv2.putText(out, text, (120, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(123, 222, 238), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(out, 'class:' + str(idx), (5, 18), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(234, 200, 100), thickness=2, lineType=cv2.LINE_AA)
        output_cam.append(out)
        #acc = sum(ori == pre for ori, pre in zip(label_test, label_test_predict)) / len(label_test)
        #accuracy.append(acc)
    output_compose= image_compose(output_cam)
    return output_compose

#output = getCAM(img=target, feature_maps=out_base, weights=weight_softmax, class_idx=2)

def all_CAM(weights):
    idx = 0
    for i in range(len(data_test)):
        src = data_test[idx][:,:,::-1]
        out_base = base_model.predict(np.expand_dims(src, axis=0))
        out_base = out_base[0]
        out= getCAM(img=src, feature_maps=out_base, weights = weights, class_idx=2)
        save_img(img = out, file_name= i, file_path=results_output_path)
        idx += 1
    return None
all_CAM(weight_softmax)





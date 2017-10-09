import os
import os.path
import argparse
import openslide
import matplotlib
matplotlib.use('Agg')

import numpy as np
from pylab import *
from os import walk
from tqdm import tqdm
from PIL import Image
from os.path import join
from sklearn import metrics
from datetime import datetime
import matplotlib.pyplot as plt
from keras.models import load_model

base_path = '/atlas/home/zwpeng/paper_rebuild/camelyon/'


def get_picture(path):
    path0 = join(base_path + path)
    picture = []
    for _,_,filenames in walk(path0):
        for filename in filenames:
            file_prefix = os.path.splitext(filename)[0]
            if os.path.exists(join(path0, file_prefix + ".tif")):
                picture.append(filename)
            else:
                print("wrong files")
    return picture


def path_to_slide_test(random_choice):
    random_choice_path = join(base_path, "test/" + random_choice)
    random_mask_path = join(base_path, "train/tumor/annotation_images/" + (random_choice.split("."))[0] + "_Mask.tif")
#     print("随机选取一张测试集的路径是：", random_choice_path)
#     print("该图片对应的mask路径是：", random_mask_path)
    origin_slide = openslide.open_slide(random_choice_path)
    mask_slide = openslide.open_slide(random_mask_path)
    return origin_slide, mask_slide


# 感兴趣区域锁定函数
def locate_ROI(origin_slide,level=6):
    origin_widths,origin_heights = origin_slide.dimensions

    object_widths,object_heights = origin_slide.level_dimensions[level]

    rgb_list_y = list()
    rgb_list_x = list()
    rgb_var_x = []
    rgb_var_y = []
    rgb_var_xi = []
    rgb_var_yi = []

    # 寻找有效区域的y值、高度
    for k in range(100):
        slide = origin_slide.read_region((0, k*origin_heights//100), level, (object_widths, object_heights//50)) 
        slide_arr = array(slide.convert("RGB"))
        arrR = np.mean(slide_arr[:,:,:1])
        arrG = np.mean(slide_arr[:,:,1:2])
        arrB = np.mean(slide_arr[:,:,2:3])
        rgb_list_y.append((arrR,arrG,arrB))
    for i,rgbVar in enumerate(rgb_list_y):
        rgb_var_y.append(np.var(rgbVar))
        if np.var(rgbVar)>=1:
            rgb_var_yi.append(i)

    effective_y = min(rgb_var_yi)*origin_heights//100        #有效区域的左上顶点y坐标找到了
    effective_heights = (max(rgb_var_yi)-min(rgb_var_yi))*origin_heights//100 + origin_heights//50  #有效区域的高度也出来了
#     print("有效区域的ｙ值是：%d" %effective_y, "有效区域的高度是：%d" %effective_heights)

    # 寻找有效区域的x值、宽度
    for j in range(100):
        slide = origin_slide.read_region((j*origin_widths//100, effective_y), level, 
                                          (object_widths//50, effective_heights//62))     # 循环顺序读取50宽的区域

        slide_arr = array(slide.convert("RGB"))
        arrR = np.mean(slide_arr[:,:,:1])
        arrG = np.mean(slide_arr[:,:,1:2])
        arrB = np.mean(slide_arr[:,:,2:3])
        rgb_list_x.append((arrR,arrG,arrB))
    for i,rgbVar in enumerate(rgb_list_x):
        rgb_var_x.append(np.var(rgbVar))
        if np.var(rgbVar)>=2:
            rgb_var_xi.append(i)

    effective_x = min(rgb_var_xi)*origin_widths//100        # 有效区域的左上顶点y坐标找到了
    effective_widths = (max(rgb_var_xi) - min(rgb_var_xi))*origin_widths//100 + origin_widths//50  # 有效区域的宽度也出来了
    return effective_x,effective_y,effective_widths,effective_heights


def locate_ROI_mask(mask_slide,mask_level=7):
    # level0　的尺寸
    mask_widths, mask_heights = mask_slide.dimensions
    # level7 的尺寸
    mask_level_widths, mask_level_heights = mask_slide.level_dimensions[mask_level]

    mask_level_slide = mask_slide.read_region((0, 0), mask_level, (mask_level_widths, mask_level_heights))
    mask_level_slide_gray = mask_level_slide.convert("L")
    mask_level_slide_arr = array(mask_level_slide_gray)

    mask_y, mask_x = nonzero(mask_level_slide_arr)  # 因为mask是黑白图，只需直接获得非零像素的坐标
    # mask_x, mask_y
    tumor_leftup_x = (min(mask_x)-1) * int(mask_slide.level_downsamples[mask_level])
    tumor_leftup_y = (min(mask_y)-1) * int(mask_slide.level_downsamples[mask_level])
    tumor_rightdown_x = (max(mask_x)+1) * int(mask_slide.level_downsamples[mask_level])
    tumor_rightdown_y = (max(mask_y)+1) * int(mask_slide.level_downsamples[mask_level])
    
#     print(tumor_leftup_x,tumor_leftup_y,tumor_rightdown_x,tumor_rightdown_y)
    mask_effective_widths = tumor_rightdown_x - tumor_leftup_x
    mask_effective_heights = tumor_rightdown_y - tumor_leftup_y
    
#     mask_tumor_area = ((max(mask_x)-min(mask_x)+2)*int(mask_slide.level_downsamples[mask_level]), 
#                        (max(mask_y)-min(mask_y)+2)*int(mask_slide.level_downsamples[mask_level]))
#     print(mask_tumor_area)        # mask区域的长宽
    return tumor_leftup_x,tumor_leftup_y,mask_effective_widths,mask_effective_heights


def effective_list(data_set):
    random_set = {}
#     random_choice = np.random.choice(data_set)    # 随机从数据集列表中选取一张图片路径
    for i in range(len(data_set)):
#        if data_set == train_set:
#            origin_slide, mask_slide = path_to_slide(data_set[i])
#        elif data_set == valid_set:
#            origin_slide, mask_slide = path_to_slide_valid(data_set[i])
        if data_set == test_set:
            origin_slide, mask_slide = path_to_slide_test(data_set[i])
        else:
            print("数据集加载不正确")

        try:
            [[effective_x,effective_y,effective_widths,effective_heights],
             [tumor_leftup_x,tumor_leftup_y,mask_effective_widths,mask_effective_heights]] = random_set[data_set[i]]
        except KeyError:
            effective_x,effective_y,effective_widths,effective_heights = locate_ROI(origin_slide)            
            tumor_leftup_x,tumor_leftup_y,mask_effective_widths,mask_effective_heights = locate_ROI_mask(mask_slide)
            random_set[data_set[i]] = [[effective_x,effective_y,effective_widths,effective_heights],
                                         [tumor_leftup_x,tumor_leftup_y,mask_effective_widths,mask_effective_heights]]
    return random_set


def prediction_from_model(patch, model):    
    prediction = model.predict(patch.reshape(1, 299, 299, 3))
    return prediction


test_set = get_picture("test/")    # 测试集数据文件夹
# print("测试数据集的数量有：",len(test_set),'\n',"测试数据集是：\n",test_set)
eff_test = effective_list(test_set)


def pred_save(args, step_len=32, widths=299, heights=299):
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model = load_model('../'+model_path)
    # 遍历测试集，获取真实值ground truth和预测值pred
    for i in tqdm(range(len(test_set))):
        s_image, m_image = path_to_slide_test(test_set[i])
        [[effective_x,effective_y,effective_widths,effective_heights],
         [tumor_leftup_x,tumor_leftup_y,mask_effective_widths,mask_effective_heights]] = eff_test[test_set[i]]

        src_slide = s_image.read_region((tumor_leftup_x,tumor_leftup_y),0,(mask_effective_widths,mask_effective_heights))
        src_mask = m_image.read_region((tumor_leftup_x,tumor_leftup_y),0,(mask_effective_widths,mask_effective_heights))
        
        pre_map = np.zeros((src_slide.size[0], src_slide.size[1]))
        y, pred = [], []    # 真实值和预测值元组初始化
        
        for h in tqdm(range(src_slide.size[1]//step_len)):
            for w in range(src_slide.size[0]//step_len):
                # mask的y值（ground truth值）
                mask_pat = src_mask.crop((w*step_len,  h*step_len, w*step_len+widths, h*step_len+heights))
                mask_pat_array = np.array(mask_pat.convert("L"))
                if mask_pat_array.any()!= 0:
                    y.append(1)
                else:
                    y.append(0)
                # slide的tumor预测值
                pat = src_slide.crop((w*step_len,  h*step_len, w*step_len+widths, h*step_len+heights))
                pat_array = np.array(pat.convert("RGB"))/255.
                pre_pat = prediction_from_model(pat_array, model)
                pred.append(pre_pat[0][0])
                pre_map[w*step_len:(w+1)*step_len,h*step_len:(h+1)*step_len] = pre_pat[0][0]

        fig1, ax = plt.subplots(figsize=(8,8))
        ax.imshow(np.transpose(pre_map), cmap='jet', vmin=0, vmax=1)
        fig1.savefig(model_path+'/'+'pre'+os.path.splitext(test_set[i])[0]+'.png') 
        plt.close()

        fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=None)
        fig2, ax = plt.subplots(figsize=(8, 8))
        ax.plot(tpr, [(1-fpr[i]) for i in range(fpr.shape[0])])    # 一句循环获取该array
        ax.set_xlim([0.0,1.05])
        ax.set_ylim([0.0,1.05])
        ax.set_xlabel('Sensitivity-True Positive Rate:正确率')
        ax.set_ylabel('Specificity-True Negative Rate:误判率')
        ax.set_title('ROC曲线,AUC= %s'%(metrics.auc(fpr, tpr)))
        fig2.savefig(model_path+'/'+'ROC'+os.path.splitext(test_set[i])[0]+'.png') 
        plt.close()

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--model_path',default='model3006450.h5')
    
    args = a.parse_args()
    
    start_time = datetime.now()
    pred_save(args)
    end_time = datetime.now()
    print("time consuming: %.1f minutes" % ((end_time - start_time).seconds / 60,))

import os.path
from os import walk
from os.path import join
import openslide
from pylab import *
from keras.utils.np_utils import to_categorical
import threading


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
                print("路径不对或者没有这种格式的文件")
    return picture

# 训练集、验证集、测试集严密隔离
train_set = get_picture("train/tumor/origin_images/")    # 训练集数据文件夹
valid_set = get_picture("validation/")    # 验证集数据文件夹
test_set = get_picture("test/")    # 测试集数据文件夹
mask_pictures = get_picture("train/tumor/annotation_images/")    # 所有的mask图文件夹


# 随机图片的读取函数
def path_to_slide(random_choice):
    random_choice_path = join(base_path, "train/tumor/origin_images/" + random_choice)
    random_mask_path = join(base_path, "train/tumor/annotation_images/" + (random_choice.split("."))[0] + "_Mask.tif")
#     print("随机选取一张训练集图的路径是：", random_choice_path)
#     print("该图片对应的mask路径是：", random_mask_path)
    origin_slide = openslide.open_slide(random_choice_path)
    mask_slide = openslide.open_slide(random_mask_path)
    return origin_slide, mask_slide


def path_to_slide_valid(random_choice):
    random_choice_path = join(base_path, "validation/" + random_choice)
    random_mask_path = join(base_path, "train/tumor/annotation_images/" + (random_choice.split("."))[0] + "_Mask.tif")
#     print("随机选取一张验证集的路径是：", random_choice_path)
#     print("该图片对应的mask路径是：", random_mask_path)
    origin_slide = openslide.open_slide(random_choice_path)
    mask_slide = openslide.open_slide(random_mask_path)
    return origin_slide, mask_slide


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


# mask感兴趣区域锁定函数
def locate_ROI_mask(mask_slide, mask_level=7):
    # level0　的尺寸
    mask_widths, mask_heights = mask_slide.dimensions
    # level7 的尺寸
    mask_level_widths, mask_level_heights = mask_slide.level_dimensions[mask_level]

    mask_level_slide = mask_slide.read_region((0, 0), mask_level, (mask_level_widths, mask_level_heights))
    mask_level_slide_gray = mask_level_slide.convert("L")
    mask_level_slide_arr = array(mask_level_slide_gray)

    mask_y, mask_x = nonzero(mask_level_slide_arr)  # 因为mask是黑白图，只需直接获得非零像素的坐标
    # mask_x, mask_y
    tumor_leftup_x = (min(mask_x) - 1) * int(mask_slide.level_downsamples[mask_level])
    tumor_leftup_y = (min(mask_y) - 1) * int(mask_slide.level_downsamples[mask_level])
    tumor_rightdown_x = (max(mask_x) + 1) * int(mask_slide.level_downsamples[mask_level])
    tumor_rightdown_y = (max(mask_y) + 1) * int(mask_slide.level_downsamples[mask_level])

    #     print(tumor_leftup_x,tumor_leftup_y,tumor_rightdown_x,tumor_rightdown_y)
    mask_effective_widths = tumor_rightdown_x - tumor_leftup_x
    mask_effective_heights = tumor_rightdown_y - tumor_leftup_y

    #     mask_tumor_area = ((max(mask_x)-min(mask_x)+2)*int(mask_slide.level_downsamples[mask_level]),
    #                        (max(mask_y)-min(mask_y)+2)*int(mask_slide.level_downsamples[mask_level]))
    #     print(mask_tumor_area)        # mask区域的长宽
    return tumor_leftup_x, tumor_leftup_y, mask_effective_widths, mask_effective_heights


# 感兴趣区域点和宽高的存放函数
def effective_list(data_set):
    random_set = {}
#     random_choice = np.random.choice(data_set)    # 随机从数据集列表中选取一张图片路径
    for i in range(len(data_set)):
        if data_set == train_set:
            origin_slide, mask_slide = path_to_slide(data_set[i])
        elif data_set == valid_set:
            origin_slide, mask_slide = path_to_slide_valid(data_set[i])
        elif data_set == test_set:
            origin_slide, mask_slide = path_to_slide_test(data_set[i])
        else:
            print("数据集加载不正确")
            return 0

        try:
            [[effective_x, effective_y, effective_widths, effective_heights],
             [tumor_leftup_x, tumor_leftup_y, mask_effective_widths, mask_effective_heights]] = random_set[data_set[i]]
        except KeyError:
            effective_x, effective_y, effective_widths, effective_heights = locate_ROI(origin_slide)
            tumor_leftup_x, tumor_leftup_y, mask_effective_widths, mask_effective_heights = locate_ROI_mask(mask_slide)
            random_set[data_set[i]] = [[effective_x, effective_y, effective_widths, effective_heights],
                                         [tumor_leftup_x, tumor_leftup_y, mask_effective_widths, mask_effective_heights]]
    return random_set


class ThreadsafeIter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    def g(*args, **kw):
        return ThreadsafeIter(f(*args, **kw))
    return g

eff_train = effective_list(train_set)
eff_valid = effective_list(valid_set)    # 所有验证数据的有效区域　存放字典
eff_test = effective_list(test_set)


@threadsafe_generator
def data_generator(data_set, batch=32, widths=256, heights=256):
    if data_set == train_set:
        random_set = eff_train
    elif data_set == valid_set:
        random_set = eff_valid
    elif data_set == test_set:
        random_set = eff_test
    else:
        print("数据集加载不正确")
        return 0

    while True:
        masks = []
        images = []
        for i in range(batch):
            random_choice = np.random.choice(data_set)

            [[effective_x, effective_y, effective_widths, effective_heights],
             [tumor_leftup_x, tumor_leftup_y, mask_effective_widths, mask_effective_heights]] = random_set[
                random_choice]

            if data_set == train_set:
                origin_slide, mask_slide = path_to_slide(random_choice)
            elif data_set == valid_set:
                origin_slide, mask_slide = path_to_slide_valid(random_choice)
            elif data_set == test_set:
                origin_slide, mask_slide = path_to_slide_test(random_choice)

            random_num = np.random.random(1)
            #             print(random_num)
            if random_num > 0.5:
                # 定义随机坐标,一定要取到一张含有tumor的图片
                random_x = np.random.randint(tumor_leftup_x, tumor_leftup_x + mask_effective_widths - widths)
                random_y = np.random.randint(tumor_leftup_y, tumor_leftup_y + mask_effective_heights - heights)
                #                 print("取tumor随机点坐标是：%d,%d"%(random_x,random_y))
                random_img_mask = mask_slide.read_region((random_x, random_y), 0, (widths, heights))
                random_img_mask_arr = array(random_img_mask.convert("L"))
                random__img_y, random_img_x = nonzero(random_img_mask_arr)
                while len(random_img_x) == 0:
                    random_x = np.random.randint(tumor_leftup_x, tumor_leftup_x + mask_effective_widths - widths)
                    random_y = np.random.randint(tumor_leftup_y, tumor_leftup_y + mask_effective_heights - heights)
                    #                     print("取tumor随机点坐标是：%d,%d"%(random_x,random_y))
                    random_img_mask = mask_slide.read_region((random_x, random_y), 0, (widths, heights))
                    random_img_mask_arr = array(random_img_mask.convert("L"))
                    random__img_y, random_img_x = nonzero(random_img_mask_arr)

                # *********************上面这个 while 循环结束后，就产生了一个合格的坐标点*********************#
                random_img = origin_slide.read_region((random_x, random_y), 0, (widths, heights))

                # ***接下来就给他贴标签，并处理成训练所需的数据结构***#
                random_img_arr = array(random_img.convert("RGB"))
                #                 x = np.expand_dims(random_img_arr, axis=0)/255.
                #             y = to_categorical(0,2)

                images.append(random_img_arr)
                mask = (random_img_mask_arr > 0).astype(int)
                masks.append(mask)

            else:
                # 定义随机坐标，一定要取到一张不含有tumor的normal图片
                random_x = np.random.randint(effective_x, effective_x + effective_widths - widths)  # 大图上,nomal有效区的起点和终点
                random_y = np.random.randint(effective_y, effective_y + effective_heights - heights)
                #                 print("取normal随机点坐标是：%d,%d"%(random_x,random_y))
                random_img_mask = mask_slide.read_region((random_x, random_y), 0, (widths, heights))
                random_img_mask_arr = array(random_img_mask.convert("L"))
                random__img_y, random_img_x = nonzero(random_img_mask_arr)
                random_img = origin_slide.read_region((random_x, random_y), 0, (widths, heights))

                #                 print("随机情况",len(random_img_x),(array(random_img.convert("RGB"))).std())

                while (array(random_img.convert("RGB"))).std() < 20.0:
                    random_x = np.random.randint(effective_x, effective_x + effective_widths - widths)
                    random_y = np.random.randint(effective_y, effective_y + effective_heights - heights)
                    #                 print("取normal随机点坐标是：%d,%d" %(random_x,random_y))
                    random_img_mask = mask_slide.read_region((random_x, random_y), 0, (widths, heights))
                    random_img_mask_arr = array(random_img_mask.convert("L"))
                    random__img_y, random_img_x = nonzero(random_img_mask_arr)
                    random_img = origin_slide.read_region((random_x, random_y), 0, (widths, heights))

                # print("颜色检测情况",len(random_img_x),(array(random_img.convert("RGB"))).std())

                while len(random_img_x) != 0:
                    random_x = np.random.randint(effective_x, effective_x + effective_widths - widths)
                    random_y = np.random.randint(effective_y, effective_y + effective_heights - heights)
                    #                 print("取normal随机点坐标是：%d,%d" %(random_x,random_y))
                    random_img_mask = mask_slide.read_region((random_x, random_y), 0, (widths, heights))
                    random_img_mask_arr = array(random_img_mask.convert("L"))
                    random__img_y, random_img_x = nonzero(random_img_mask_arr)
                    random_img = origin_slide.read_region((random_x, random_y), 0, (widths, heights))

                # print("非tumor区检测情况",len(random_img_x), (array(random_img.convert("RGB"))).std())

                # *********************上面这个 while 循环结束后，就产生了一个合格的坐标点*********************#
                random_img = origin_slide.read_region((random_x, random_y), 0, (widths, heights))

                # ***接下来就给他贴标签，并处理成训练所需的数据结构***#
                random_img_arr = array(random_img.convert("RGB"))
                #                 x = np.expand_dims(random_img_arr, axis=0)/255.
                #             y = to_categorical(1,2)

                images.append(random_img_arr)
                mask = np.zeros((widths, heights))
                masks.append(mask)
        X_train = np.array(images)
        y_train = np.array(masks)
        y_train = to_categorical(y_train, num_classes=2).reshape(y_train.shape[0], widths, heights, 2)
        yield X_train, y_train







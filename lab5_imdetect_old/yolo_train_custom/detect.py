"""
Created on Sat Apr 10 13:14:08 2021

@author: Acerjunior
"""

from models import *
from utils.utils import *
from utils.datasets import *

import os
import time
import datetime
import numpy as np

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

# задаем каталоги и файлы
image_folder = "data\samples" # каталог с тестовыми изображениями
model_config = "config/yolov3.cfg" # файл с конфигурацией детектор
data_config = "config/coco.data"    # файл с описанием набора данных
weights = "config/yolov3_9.weights" # файл весов
class_path = "config/coco.names"         # файл с именами классов


# определяем доступна ли CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# создаем каталог для сохранения результатов
os.makedirs("data\output", exist_ok=True)

# Инициализируем модель и загружаем в нее предобученные веса
model = Darknet(model_config).to(device)
model.load_weights(weights)

if device.type == 'cuda':
    model = model.cuda()

# переключаем модель в режим предсказания
model.eval() 

# задаем параметры и создаем dataloader
batch_size = 1                         # размер батча
n_cpu = 2           # количество процессов для загрузи и преобразования данных
img_size = 416      # размер изображения
dataloader = DataLoader(
    ImageFolder(image_folder, img_size=img_size),
    batch_size=batch_size,
    shuffle=False,
    num_workers=n_cpu,
)

# считываем имена классов
classes = load_classes(class_path)
# парсим файл конфигурации
data_config = parse_data_config(data_config)
num_classes = int(data_config["classes"])


# задаем тип тензора в зависимости от доступности CUDA
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# ОБРАБОТКА ИЗОБРАЖЕНИЙ
conf_thres = 0.99                      # порог для отбора ограничивающих рамок 
nms_thres = 0.1                        # порог non-maximum suppression

imgs = [] 
img_detections = []

# последовательно подаем изображения на детектор
prev_time = time.time()
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    input_imgs = Variable(input_imgs.type(Tensor))
    # оставляем только то, что выше заданных порогов 
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, num_classes, conf_thres, nms_thres)

    # вывод диагностической информации
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print("\t+ Batch %d, Time: %s" % (batch_i, inference_time))

    # сохраняем изображения и обнаруженные рамки
    imgs.extend(img_paths)
    img_detections.extend(detections)

# задаем цвета рамок bbox разные для разных классов
cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

# Выводим на экран результаты и сохраняем их в файл
for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

    print("(%d) Image: '%s'" % (img_i, path))

    img = np.array(Image.open(path))
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    filename = path.split("\\")[-1].split(".")[0]
    file_name_res = open(f'data\output\{filename}' + '.txt', 'w')

    # отрисовываем изображения и сохраняем их
    if detections is not None:
        # изменяем размер bboxes под размер изображения 
        detections = rescale_boxes(detections, img_size, img.shape[:2])
        (height, weight) = img.shape[:2] 
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

            box_w = x2 - x1
            box_h = y2 - y1
            
            file_name_res.write("0 " + str(round((float((x1 + box_w/2)/weight)), 6)) + " " + str(round((float((y1 + box_h/2)/height)), 6)) + " " +
                                str(round((float(box_w/weight)), 6)) + " " + str(round((float(box_h/height)),6)) + "\n")

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # рисуем прямоугольники и подписи
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(bbox)
            plt.text(
                x1,
                y1,
                s=classes[int(cls_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("\\")[-1].split(".")[0]
        plt.savefig(f"data\output\{filename}.png", bbox_inches="tight", pad_inches=0.0)
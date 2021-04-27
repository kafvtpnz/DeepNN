"""
Created on Sat Apr 10 13:14:07 2021

@author: Acerjunior
"""

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import torch
from torch.autograd import Variable

# задаем каталоги
model_config ="config/yolov3.cfg"   # файл с конфигурацией детектора
data_config = "config/coco.data"    # файл с описанием набора данных
weights = "config/yolov3.weights"   # файл весов
class_names ="config/coco.names"    # файл с именами классов
checkpoint_dir = "checkpoints" # каталог для сохранения весов

# определяем доступна ли CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# создаем каталог для промежуточного сохранения весов
os.makedirs(checkpoint_dir, exist_ok=True)

# считываем имена классов
classes = load_classes(class_names)

# считываем параметры данных
data_config = parse_data_config(data_config)
train_path = data_config["train"]


# Инициализируем модель и загружаем в нее предобученные веса
model = Darknet(model_config)
model.load_weights(weights)
#model.apply(weights_init_normal) # инициализация весов для "чистого" обучения

if device.type == 'cuda':
    model = model.cuda()

# переключаем модель в режим обучения
model.train()

# Задаем гиперпараметры, часть считываем из файла конфигурации
epochs = 2 # количество эпох
checkpoint_interval = 1 # количество эпох, через которое сохраняются веса

# парсим файл конфигурации
hyperparams = parse_model_config(model_config)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])

# Создаем dataloader
n_cpu = 1 # количество процессов для загрузи и преобразования данных
batch_size = int(hyperparams["batch"]) # размер батча

dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=batch_size, shuffle=False, num_workers=n_cpu)


# задаем тип тензора в зависимости от доступности CUDA
Tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor

# задаем тип и параметры оптимизатора
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=learning_rate, momentum=momentum, weight_decay=decay)

# запускаем цикл обучения
for epoch in range(epochs):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()
        # прямой проход с вычислением ошибок
        loss = model(imgs, targets)
        # обратный проход и корректировка весов
        loss.backward()
        optimizer.step()
        
        # вывод диагностической информации
        print(
            "[Эпоха %d из %d, Batch %d из %d] [Потери по: x %f, y %f, w %f, h %f, cls %f, total %f. Точность: recall - %.5f, precision - %.5f]"
            % (
                epoch,
                epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )

        model.seen += imgs.size(0)
    # сохранение весов
    if epoch % checkpoint_interval == 0:
        model.save_weights("%s/yolov3_%d.weights" % (checkpoint_dir, epoch))

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:05:20 2021

@author: AM4
"""
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# Сначала определим на каком устройстве будем работать - GPU или CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Затем загружаем данные (взяты отсюда https://download.pytorch.org/tutorial/hymenoptera_data.zip)

batch_size = 10

# Так как сеть, которую мы планируем взять за базу натренирована на изображениях 
# определенного размера, то наши изображения необходимо к ним преобразовать
data_transforms = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225] )
    ])

# загрузим отдельно обучающий набор
train_dataset = torchvision.datasets.ImageFolder(root='./hymenoptera_data/train', 
                                              transform=data_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=True,  num_workers=2)

# посмотрим какие классы содержатся в наборе
train_dataset.classes

# сохраним названия этих классов
class_names = train_dataset.classes

# посмотрим на размер нашего набора данных
len(train_dataset.samples)


# и отдельно загрузим тестовый набор
test_dataset = torchvision.datasets.ImageFolder(root='./hymenoptera_data/val',
                                             transform=data_transforms)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                    shuffle=True, num_workers=2) 

# проверим, что в наборе данные получились требуемого размера
inputs, classes = next(iter(train_loader))
inputs.shape


# построим на картинке
img = torchvision.utils.make_grid(inputs, nrow = 5) # метод делает сетку из картинок
img = img.numpy().transpose((1, 2, 0)) # транспонируем для отображения в картинке
plt.imshow(img)



# в качестве донора возьмем преобученную на ImageNet наборе сеть AlexNet
net = torchvision.models.alexnet(pretrained=True)

# можно посмотреть структуру этой сети
# print(net)

# так как веса feature_extractor уже обучены, нам нужно их заморозить, чтобы 
# быстрее научился наш классификатор
#  для этого отключаем слоев (включая слои feature_extractor-а) градиенты
for param in net.parameters():
    param.requires_grad = False

# так как выходной слой AlexNet содержит 1000 нейронов (по количеству классов в ImageNet)
# то нам нужно его заменить на слой, содержащий только 2 класса

num_classes = 2

new_classifier = net.classifier[:-1] # берем все слой классификатора кроме последнего
new_classifier.add_module('fc',nn.Linear(4096,num_classes))# добавляем последним стоем новый
net.classifier = new_classifier # меняем классификатор сети

net = net.to(device)

# проверим эффективность новой сети
correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad(): # отключим вычисление граиентов, т.к. будем делать только прямой проход
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images) # делаем предсказание по пакету
        _, pred_class = torch.max(pred.data, 1) # выбираем класс с максимальной оценкой
        correct_predictions += (pred_class == labels).sum().item()

print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')
# явно требуется обучение


# Перейдем к обучению
# Зададим количество эпох обучения, функционал потерь и оптимизатор
num_epochs = 2
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# создаем цикл обучения и замеряем время его выполнения
import time
t = time.time()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # прямой проход
        outputs = net(images)
        # вычисление значения функции потерь
        loss = lossFn(outputs, labels)
         # Обратный проход (вычисляем градиенты)
        optimizer.zero_grad()
        loss.backward()
        # делаем шаг оптимизации весов
        optimizer.step()
        # выводим немного диагностической информации
        if i%100==0:
            print('Эпоха ' + str(epoch) + ' из ' + str(num_epochs) + ' Шаг ' +
                  str(i) + ' Ошибка: ', loss.item())

print(time.time() - t)


# Еще раз посчитаем точность нашей модели
correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad(): # отключим вычисление граиентов, т.к. будем делать только прямой проход
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images) # делаем предсказание по пакету
        _, pred_class = torch.max(pred.data, 1) # выбираем класс с максимальной оценкой
        correct_predictions += (pred_class == labels).sum().item()

print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')
# уже лучше


# Реализуем отображение картинок и их класса, предсказанного сетью
inputs, classes = next(iter(test_loader))
pred = net(inputs.to(device))
_, pred_class = torch.max(pred.data, 1)

for i,j in zip(inputs, pred_class):
    img = i.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(class_names[j])
    plt.pause(2)



# Нашу модель можно сохранить в файл для дальнейшего использования
torch.save(net.state_dict(), 'CnNet.ckpt')

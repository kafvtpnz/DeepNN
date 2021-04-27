import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# Сначала определим на каком устройстве будем работать - GPU или CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Затем загружаем данные
# Работа с данными в pytorch осуществляется через класс datasets 
# Этот класс содержит основную информацию об используемом наборе данных 
# пути расположения файлов, метки классов, разметки и аннотации
# для работы с собственными данными нужно создать свой dataset из каталога с
# файлами или можно воспользоваться одним из предустановленных наборов данных
# полный список имеющихся dataset-ов можно посмотреть по ссылке 
# https://pytorch.org/vision/stable/datasets.html

# загрузим набор MNIST - набор для распознавания рукописных символов
# в параметрах укажем каталог где расположен (или будет храниться набор),
# Параметр train указывает какую часть набора будем загружать (True - обучающую, False - тестовую)
# Параметр transform указывает какие преобразования с данными необходимо проделать
# это могут быть обрезка, повороты, изменения яркасти и др. (тут явно укажем, что сразу
# изображение необходимо преобразовать в тензор)
# Параметр download указывает на необходимость загрузки набора данных через интернет,
# усли он не был найден в указанном каталоге

# загрузим отдельно обучающий набор
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)
# и отдельно тестовый набор
test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# посмотрим какие классы содержатся в наборе
train_dataset.classes

# сохраним названия этих классов
class_names = train_dataset.classes

# Данные из набора можно загрузить в тензор
train_set = train_dataset.data

# посмотрим на размер нашего набора данных
train_set.shape

# хотелось бы посмотреть что мы загрузили, для этого нужно преобразовать 
# тензор в массив numpy
nptrain_set = train_set.numpy()

# теперь можно отобразить его составляющие картинкой
plt.imshow(nptrain_set[10000,:,:])


# В процессе обучения информация подается пакетами (batch), 
# для их формирования предназначен класс DataLoader
# который является итерируемым (т.е. каждый вызов его итерации изменяет состояние) 
# формирует пакет размером batch_size из случайно выбранных (shuffle=True)
# или последовательно
# также поддерживается многопоточная загрузка данных (num_workers=nThreads)

batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
    

# каждая итерация обращения к созданному DataLoader возвращает batch данных и их классы
inputs, classes = next(iter(train_loader))

inputs.shape

classes

# построим изображение из пакета
img = torchvision.utils.make_grid(inputs) # метод делает сетку из картинок
img.numpy().shape

img = img.numpy().transpose((1, 2, 0)) # транспонируем для отображения в картинке
img.shape
plt.imshow(img)


# Теперь можно переходить к созданию сети
# Для этого будем использовать как и ранее метод Sequential
# который объединит несколько слоев в один стек
class CnNet(nn.Module):
    def __init__(self, num_classes=10):
        nn.Module.__init__(self)
        self.layer1 = nn.Sequential(
        # первый сверточный слой с ReLU активацией и maxpooling-ом
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # второй сверточный слой с ReLU активацией и maxpooling-ом
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # классификационный слой
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1) # флаттеринг
        out = self.fc(out)
        return out


# Зададим количество эпох обучения (каждая эпоха прогоняет обучающий набор 1 раз)
num_epochs = 2
num_classes = 10

# создаем экземпляр сети
net = CnNet(num_classes).to(device)

# Задаем функцию потерь и алгоритм оптимизации
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

# посчитаем точность нашей модели: количество правильно классифицированных цифр
# поделенное на общее количество тестовых примеров

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

# Нашу модель можно сохранить в файл для дальнейшего использования
torch.save(net.state_dict(), 'CnNet.ckpt')


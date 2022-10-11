import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers


#Определим некоторые переменные
#Использование tensorflow в качестве бэкграунда
os.environ["KERAS_BACKEND"] = "tensorflow"
#Проверяем, что можем воспроизвести эксперимент и получить такие же результаты
np.random.seed(10)
#Задаем размер вектора шума
random_dim = 100


# Cобраем данные и делаем их предварительную обработк. 
# Будем использовать датасет MNIST (набор изображений цифр от 0 до 9)
# Генерация символов с gan
# Пример символов из датасета MNIST


def load_minst_data():
    # Загружаем данные
    # Команда mnist.load_data() является частью Keras и позволяет импортировать датасет в рабочее пространство
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Нормализуем ввод в диапазоне [-1, 1] 
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    # преобразуем x_train с (60000, 28, 28) к (60000, 784) 
    # имеем 784 столбца за строку
    x_train = x_train.reshape(60000, 784)
    return (x_train, y_train, x_test, y_test)


# Cоздаем сети генератора и дискриминатора 
# Для обеих сетей используем оптимизатор Adam
# В обоих случаях сеть будет состоять из трех скрытых слоев с активационной функцией Leaky Relu 
# Также следует добавить в дискриминатор dropout слои, чтобы улучшить его надежность, качество (robustness) на изображениях, которые не были показаны


def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(784, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator

def get_discriminator(optimizer):
    discriminator = Sequential()
    discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator

# Cоединяем генератор с дискриминатором

def get_gan_network(discriminator, random_dim, generator, optimizer):
    # устанавливаем обучение в ложь, тк хотим только обучать 
    # генератор или дискриминатор за раз
    discriminator.trainable = False
    # входной шум GAN 
    gan_input = Input(shape=(random_dim,))
    # выход генератора (изображение)
    x = generator(gan_input)
    # ваход дискриминатора (вероятность)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan

# Дополнительно создаем функцию, сохраняющую сгенерированные изображения через каждые 20 эпох 
# Этот шаг не является основным в туториале, вам не обязательно полностью понимать выводящую изображение функцию.

def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)

# Обучаем нейросеть и смотрим на результаты — изображения

def train(epochs=1, batch_size=128):
    # получаем данные тренировки и тестирования
    x_train, y_train, x_test, y_test = load_minst_data()
    # разделяем данные обучения на пакеты размером 128
    batch_count = x_train.shape[0] / batch_size

    # создаем сеть
    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, random_dim, generator, adam)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15, end=" ")
        for _ in tqdm(range(int(batch_count))):
            # получаем случайный набор помех на входе изображения
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # создаем поддельные изображения MNIST
            generated_images = generator.predict(noise)
            X = np.concatenate([image_batch, generated_images])

            # метки для сгенерированных и реальных данных
            y_dis = np.zeros(2*batch_size)
            # одностороннее выравнивание метки
            y_dis[:batch_size] = 0.9

            # обучаем дискриминатор
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # обучаем генератор
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 5 == 0:
            plot_generated_images(e, generator)

if __name__ == '__main__':
    train() # 1 up me to 20
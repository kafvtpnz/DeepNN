# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 21:29:39 2021

@author: AM4
"""
# ПЕРЕД запуском установить пакеты opencv и scikit-image
# conda install -c conda-forge py-opencv
# conda install scikit-image
# !!! при возникновении проблем с установкой opencv (при импорте ругается на отсутствие DLL)
# в windows7 необходимо установить его командой pip install opencv-python 


# ПЕРЕД НАЧАЛОМ РАБОТЫ !!! проверить установку текущего каталога

# импортируем модули
import torch
import cv2
import torchvision.transforms as transforms
import time


# пара функций необходимых для запуска на CPU
def checkpoint_from_distributed(state_dict):
    """
    Функция проверяет была ли сеть обучена с использованием  DistributedDataParallel (используется на кластерах GPU)
    Такая сеть не может быть запущена на CPU
    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret

def unwrap_distributed(state_dict):
    """
    Функция преобразует сеть, обученную с использованием  DistributedDataParallel (используется на кластерах GPU)
    к виду, используемому для работы на CPU
    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.1.', '')
        new_key = new_key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict

# функция для отрисовки прямоугольников на изображении
def draw_bboxes(image, results, classes_to_labels):
    for image_idx in range(len(results)):
        # размеры изображения
        orig_h, orig_w = image.shape[0], image.shape[1]
        # достаем координаты, название класса и скор
        bboxes, classes, confidences = results[image_idx]
        for idx in range(len(bboxes)):
            # координаты углов
            x1, y1, x2, y2 = bboxes[idx]
            # resize под изображение
            x1, y1 = int(x1*300), int(y1*300)
            x2, y2 = int(x2*300), int(y2*300)
            x1, y1 = int((x1/300)*orig_w), int((y1/300)*orig_h)
            x2, y2 = int((x2/300)*orig_w), int((y2/300)*orig_h)
            # рисуем прямоугольник
            cv2.rectangle(
                image, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA
            )
            # подписываем название класса
            cv2.putText(
                image, classes_to_labels[classes[idx]-1], (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
    return image


# Делаем необходимые приготовления

# определяем доступные устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# задаем необходимые преобразования изображений под SSD
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
    
# загружаем модель и утилиты (первый раз загружается через интернет, поэтому необходимо подключение)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

if device.type == 'cpu':
    ssdetector = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', pretrained=False)
    checkpoint = torch.hub.load_state_dict_from_url('https://api.ngc.nvidia.com/v2/models/nvidia/ssd_pyt_ckpt_amp/versions/19.09.0/files/nvidia_ssdpyt_fp16_190826.pt', map_location="cpu")
    checkpoint = checkpoint['model']
    if checkpoint_from_distributed(checkpoint):
        checkpoint = unwrap_distributed(checkpoint)
    ssdetector.load_state_dict(checkpoint)
elif 'cuda':
    ssdetector = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', pretrained=True)

ssdetector.to(device) # отправляем модель на устройство CPU или GPU
ssdetector.eval()     # переключаем модель в режим предсказания (это нужно т.к. некоторые слои меняют свое поведение в режиме обучения и предсказания)


# Теперь непостредственно работа
video_path = 'input\\video_2.mp4' # задаем путь к файлу

# открываем видеофайл
vid = cv2.VideoCapture(video_path)
if (vid.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# определяем размеры кадра
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))

# задаем выходной файл для сохранения результата
file_name = video_path.split('\\')[-1].split('.')[0]
out = cv2.VideoWriter(f"outputs/{file_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 20, 
                      (frame_width, frame_height))
f_count = 0 # счетчик кадров
total_fps = 0 # кадры в секунду

# обрабатываем видео покадрово
while(vid.isOpened()):
    # берем кадр
    ret, frame = vid.read()
    if ret == True:
        frame_tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # изменяем цветовую палитру иприменяем трансформации
        # добавляем размерность, т.к. сети в pytorch работают только с батчами
        frame_tensor = frame_tensor.unsqueeze(0).to(device)

        # начинаем считать время, затрачиваемое на обработку
        start_time = time.time()
        
        # делаем предсказание
        with torch.no_grad():
            detections = ssdetector(frame_tensor)
       # декодируем результат
        results_per_input = utils.decode_results(detections)
        # отбираем все предсказания для которых "уверенность" алгоритма больше порого
        threshold = 0.5
        best_results_per_input = [utils.pick_best(results, threshold) for results in results_per_input]
        # по метке класса восстанавливаем его название
        classes_to_labels = utils.get_coco_object_dictionary()
        # отрисовываем ограничивающий прямоугольник, и название класса на картинке
        frame_result = draw_bboxes(frame, best_results_per_input, classes_to_labels)
    
        end_time = time.time()
        # считаем fps
        fps = 1 / (end_time - start_time)
        total_fps += fps
        
       
        # добавляем fps на изображение
        cv2.putText(
            frame_result, f"{fps:.3f} FPS", (5, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
            2
        )
        
        # отображаем результат на экране
        cv2.imshow('image', frame_result)
        wait_time = max(1, int(fps/4))
        out.write(frame_result)
        # жмем `q` для выхода из цикла
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    else:
        break
    
# закрываем файл VideoCapture()
vid.release()
out.release()
# cv2.destroyAllWindows()

# Считаем средний fps
avg_fps = total_fps / f_count
print(f"Average FPS: {avg_fps:.3f}")
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:27:20 2024

@author: AM4
"""

import ultralytics
from ultralytics import YOLO
import cv2

# проверяем что доступно из оборудования
ultralytics.checks()

# Создаем модель. При первом вызове загружает веса, требуется интернет
model = YOLO("yolov8s.pt")

# Запускаем модель. На вход подаем изображения с диска, указав к нему путь
results = model("image_1.jpg")

# Достаем результаты модели
result = results[0]

# Выводим результаты на экран при помощи OpenCV
cv2.imshow("YOLOv8", result.plot())


# В результатах содержится много полезной информации:
boxes = result.boxes.cpu()       # Рамки объектов по умолчанию в формате YOLO
print(boxes)
boxes = result.boxes.xyxy.cpu()  # Координаты рамок можно преобразовать в пиксели
print(boxes)

# Вероятности для каждого из обнаруженных объектов
# чем больше, тем сеть увереннее что это именно этот объект
confidences = result.boxes.conf
print(confidences)

classes = result.boxes.cls # Номера классов для каждого объекта
print(classes)
class_names = result.names # Сами названия классов  
print(class_names)

# Даже исходное изображение
img = result.orig_img


# напишем свою функцию для отрисовки прямоугольников на изображении
def draw_bboxes(image, results):
    boxes = results[0].boxes.cpu()
    orig_h, orig_w = results[0].orig_shape # размеры изображения
    class_names = results[0].names             # названия классов  
    for box in boxes:    
        # достаем координаты, название класса и скор
        class_idx = box.cls
        confidence = box.conf
        
        # Будем отрисовывать только то в чем сеть хорошо уверена
        if confidence>0.7:
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            # рисуем прямоугольник
            cv2.rectangle(
                image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2, cv2.LINE_AA
            )
            
            # подписываем название класса
            cv2.putText(
                image, class_names[class_idx.item()], (int(x1), int(y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
        
    return image

# вызовем функцию и выведем на экран то чот получилось
annotated_img = draw_bboxes(img.copy(), results)
cv2.imshow("YOLOv8", annotated_img)


# Теперь попробуем обучить на собственном датасете
# он доступен по ссылке https://drive.google.com/file/d/1qS_yGj3vkmEuv9Fc3Xffqg6fE5C8m3AG/view?usp=drive_link

# перед обучением необходимо скорректировать пути в файле masked.yaml
# пути должны быть абсолютными
# обучение может занять много времени, особенно на CPU
results = model.train(data="masked.yaml", model="yolov8s.pt", epochs=1, batch=8,
                      project='masks', val = True, verbose=True)

results = model("maksssksksss0.JPG")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())


# Попробуем обработать видео

# Открываем видеофайл
video_path = "masktrack.mp4"
cap = cv2.VideoCapture(video_path)


while cap.isOpened():
    # Считываем кадр
    success, frame = cap.read()

    if success:
        # Если кадр прочитался успешно, запускаем модель
        results = model(frame)
        
        result = results[0]
        cv2.imshow("YOLOv8", result.plot())

        # По нажатию "q" будем выходить из цикла
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Освобождаем поток видео и закрываем окно отображения
cap.release()
cv2.destroyAllWindows()
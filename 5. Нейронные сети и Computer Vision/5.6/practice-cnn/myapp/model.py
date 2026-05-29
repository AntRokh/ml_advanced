import os
import numpy as np
import torch
import torch.nn as nn
import cv2


# Определяем модель
class CNN(nn.Module):
    def __init__(self, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, out_features=n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 7 * 7 * 64)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

class Model:
    #model_path='model.ckpt'
    def __init__(self, model_path='model.ckpt'):
        # Загрузить модель из файла
        state_dict = torch.load(model_path)

        # Создать экземпляр модели
        self.model = CNN(n_classes=47) # Замените CNN на вашу модель

        # Загрузить веса модели
        self.model.load_state_dict(state_dict)

        # Перевести модель в режим оценки
        self.model.eval()

        # Переместить модель на устройство (CPU или GPU)
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Создаем словарь соответствий из файла emnist-balanced-mapping.txt.
        mapping_path = os.path.join('emnist-balanced-mapping.txt')  # Путь к файлу на уровень выше
        self.symbol_map = {}
        with open(mapping_path, 'r') as f:
            for line in f:
                label, symbol = line.strip().split()
                self.symbol_map[int(label)] = chr(int(symbol))

    def predict(self, x):
        """
        Предсказывает символ для изображения.

        Args:
            x: Массив NumPy, представляющий изображение. Формат: (28, 28).

        Returns:
            pred: Строка, представляющая предсказанный символ.
        """
        # Преобразование изображения в тензор PyTorch
        image_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

        # Предсказание класса
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = output.argmax(dim=1).item()

        # Получение символа по индексу
        pred = self.symbol_map[prediction]

        return pred
import os
import pickle
import numpy as np


class Model:
    def __init__(self):
        model_path = os.path.join('myapp', 'model.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)  # загружаем модель

        # Создаем словарь соответствий из файла emnist-balanced-mapping.txt.
        mapping_path = os.path.join('emnist-balanced-mapping.txt')  # Путь к файлу на уровень выше
        self.symbol_map = {}
        with open(mapping_path, 'r') as f:
            for line in f:
                label, symbol = line.strip().split()
                self.symbol_map[int(label)] = chr(int(symbol))


    def predict(self, x):
        '''
        Parameters
        ----------
        x : np.ndarray
            Входное изображение -- массив размера (28, 28)
        Returns
        -------
        pred : str
            Символ-предсказание 
        '''
        x = x.reshape(1, -1)
        pred_label = int(self.model.predict(x)[0])
        pred_symbol = self.symbol_map.get(pred_label, 'Не опознано')

        return pred_label, pred_symbol



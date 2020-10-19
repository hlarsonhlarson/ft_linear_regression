import os
import json
import pandas as pd
from config import learning_rate, iters_num
from config import data_folder, coefficients_filename, data_filename
from sklearn import preprocessing
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, filename, folder='data'):
        self.folder = folder
        self.df = pd.read_csv(os.path.join(self.folder, filename))
        self.df = self.df.rename(columns={'km': 'x', 'price': 'y'})
        # Нормализация входных значений.
        self.mean_x = self.df['x'].mean()
        self.std_x = self.df['x'].std()
        self.df.x = (self.df['x'] - self.df['x'].mean()) / self.df['x'].std()
        self.teta_0 = None
        self.teta_1 = None

    def show_data(self):
        X = self.df['x']
        y = self.df['y']
        plt.plot(X, self.teta_1 * X + self.teta_0)
        plt.scatter(X, y, marker='x')
        plt.xlabel('mileage')
        plt.ylabel('price')
        plt.show()

    def estimate_price(self, teta_0, teta_1):
        self.df['estimate_price'] = (self.df['x'] * teta_1) + teta_0

    def new_teta_0(self):
        result = (self.df['y'] - self.df['estimate_price']).mean()
        return result

    def new_teta_1(self):
        result = ((self.df['y'] - self.df['estimate_price']) * self.df['x']).mean()
        return result

    def searching_coefficients(self):
        self.teta_1 = 0
        self.teta_0 = 0
        for epoch in range(iters_num):
            self.estimate_price(self.teta_0, self.teta_1)
            self.teta_0 += self.new_teta_0()
            self.teta_1 += self.new_teta_1()
        return self.teta_0, self.teta_1

    def save_coefficients(self, filename='coefficients.json', folder='data'):
        self.teta_0 = self.teta_0 - self.teta_1 * self.mean_x/self.std_x
        self.teta_1 = self.teta_1 / self.std_x
        result = {'teta_0': self.teta_0, 'teta_1': self.teta_1, 'mean_x': self.mean_x, 'std_x': self.std_x}
        with open(os.path.join(folder, filename), 'w') as file:
            file.write(json.dumps(result))


if __name__ == '__main__':
    regression = LinearRegression(data_filename, folder=data_folder)
    regression.searching_coefficients()
    # regression.show_data()
    regression.save_coefficients(filename=coefficients_filename, folder=data_folder)

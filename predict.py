import os
import json
from config import data_folder, coefficients_filename, data_filename


def error(text):
    full_message = f'ERROR: {text}.'
    print(full_message)
    exit(1)

def get_coefficients():
    full_filename = os.path.join(data_folder, coefficients_filename)
    if os.path.exists(full_filename) and os.path.isfile(full_filename):
        with open(full_filename, 'r') as file:
            file = file.read()
        try:
            data = json.loads(file)
            return data['teta_0'], data['teta_1'], data['mean_x'], data['std_x']
        except:
            error(f'format of the file "{full_filename}" is incorrect (this must be json contains dict with 2 keys: "teta_0" and "teta_1", all them is floats or ints)')
    error(f'the ("{full_filename}") file with coefficients is not exists')

def predict(teta_0, teta_1, number):
    result = teta_0 + teta_1 * number
    return result

def main():
    number = float(input('Enter mileage (km): '))
    try:
        number = float(number)
    except ValueError:
        error('the mileage must be a number')
    teta_0, teta_1, mean, std = get_coefficients()
    result = predict(teta_0, teta_1, number)
    print(f'Price of the car is {result}')

if __name__ == '__main__':
    main()

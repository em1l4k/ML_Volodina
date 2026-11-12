import numpy as np

# ============================================================
# 1. НЕЙРОСЕТЬ, ОПРЕДЕЛЯЮЩАЯ ПОЛ ЧЕЛОВЕКА ПО ВЕСУ И РОСТУ
# ============================================================

'''
#Изменяем learning_rate и количество итераций.


def neural_networks(inp, weights):
    return inp.dot(weights)


def get_error(true_value, prediction_value):
    return np.sqrt(np.mean((true_value - prediction_value) ** 2))


inp = np.array([
    [150, 40],
    [140, 35],
    [155, 45],
    [185, 95],
    [145, 40],
    [195, 100],
    [180, 95],
    [170, 80],
    [160, 90]
])

true_predictions = np.array([0, 0, 0, 100, 0, 100, 100, 100, 100])

weights = np.array([0.2, 0.3])

learning_rate = 0.00001

# Нормализуем входные данные, чтобы обучение было стабильнее
inp_normalized = inp / 100

for i in range(500):
    delta = np.zeros_like(weights)

    for j in range(len(inp_normalized)):
        current_inp = inp_normalized[j]
        true_prediction = true_predictions[j]

        prediction = neural_networks(current_inp, weights)

        delta += (
            (prediction - true_prediction)
            * current_inp
            * learning_rate
        )

    weights = weights - delta / len(inp_normalized)

    predictions = neural_networks(inp_normalized, weights)
    error = get_error(true_predictions, predictions)

    print("Итерация:", i + 1)
    print("RMSE:", error)
    print("Weights:", weights)
    print("-------------------")

print("Предсказание для [150, 45]:",
      neural_networks(np.array([150, 45]) / 100, weights))

print("Предсказание для [170, 85]:",
      neural_networks(np.array([170, 85]) / 100, weights))

'''
# ============================================================
# 2. ИЗМЕНЕНИЕ СКОРОСТИ ОБУЧЕНИЯ
# ============================================================

'''
#Проверяем learning_rate:
#0.000001
#0.00001
#0.0001
#Сравниваем итоговую ошибку.


for learning_rate in [0.000001, 0.00001, 0.0001]:

    weights = np.array([0.2, 0.3])

    for i in range(500):
        delta = np.zeros_like(weights)

        for j in range(len(inp_normalized)):
            current_inp = inp_normalized[j]
            true_prediction = true_predictions[j]

            prediction = neural_networks(current_inp, weights)

            delta += (
                (prediction - true_prediction)
                * current_inp
                * learning_rate
            )

        weights = weights - delta / len(inp_normalized)

    predictions = neural_networks(inp_normalized, weights)

    print("Learning rate:", learning_rate)
    print("RMSE:", get_error(true_predictions, predictions))
    print("Weights:", weights)
    print("-------------------")

'''
# ============================================================
# 3. ИЗМЕНЕНИЕ КОЛИЧЕСТВА ИТЕРАЦИЙ
# ============================================================


#Проверяем 100, 500 и 10000 итераций.

def neural_networks(inp, weights):
    return inp.dot(weights)


def get_error(true_value, prediction_value):
    return np.sqrt(np.mean((true_value - prediction_value) ** 2))


inp = np.array([
    [150, 40],
    [140, 35],
    [155, 45],
    [185, 95],
    [145, 40],
    [195, 100],
    [180, 95],
    [170, 80],
    [160, 90]
])

true_predictions = np.array([0, 0, 0, 100, 0, 100, 100, 100, 100])

weights = np.array([0.2, 0.3])

learning_rate = 0.00001

# Нормализуем входные данные, чтобы обучение было стабильнее
inp_normalized = inp / 100

for epochs in [100, 500, 10000]:

    weights = np.array([0.2, 0.3])
    learning_rate = 0.00001

    for i in range(epochs):
        delta = np.zeros_like(weights)

        for j in range(len(inp_normalized)):
            current_inp = inp_normalized[j]
            true_prediction = true_predictions[j]

            prediction = neural_networks(current_inp, weights)

            delta += (
                (prediction - true_prediction)
                * current_inp
                * learning_rate
            )

        weights = weights - delta / len(inp_normalized)

    predictions = neural_networks(inp_normalized, weights)

    print("Количество итераций:", epochs)
    print("RMSE:", get_error(true_predictions, predictions))
    print("Weights:", weights)
    print("-------------------")


# ============================================================
# 4. ИЗМЕНЕНИЕ ФУНКЦИИ ОШИБКИ НА RMSE
# ============================================================


#Используем RMSE для вычисления ошибки.

'''

def get_error(true_value, prediction_value):
    return np.sqrt(np.mean((true_value - prediction_value) ** 2))


weights = np.array([0.2, 0.3])
learning_rate = 0.00001

for i in range(500):
    delta = np.zeros_like(weights)

    for j in range(len(inp_normalized)):
        current_inp = inp_normalized[j]
        true_prediction = true_predictions[j]

        prediction = neural_networks(current_inp, weights)

        delta += (
            (prediction - true_prediction)
            * current_inp
            * learning_rate
        )

    weights = weights - delta / len(inp_normalized)

    predictions = neural_networks(inp_normalized, weights)

    print("Итерация:", i + 1)
    print("RMSE:", get_error(true_predictions, predictions))
    print("-------------------")


# ============================================================
# 5. НЕЙРОСЕТЬ, КОТОРАЯ СУММИРУЕТ ДВА ЧИСЛА
# ============================================================


#Вход:
#[10, 5] -> 15
#[0, -5] -> -5
#[2, 6] -> 8

#Проверка:
#[12, 4] -> 16
#[3, -8] -> -5


def neural_networks(inp, weights):
    return inp.dot(weights)
def get_error(true_value, prediction_value):
    return np.sqrt(np.mean((true_value - prediction_value) ** 2))

inp = np.array([
    [10, 5],
    [0, -5],
    [2, 6]
])

true_predictions = np.array([15, -5, 8])

weights = np.array([0.5, 0.5])

learning_rate = 0.01

for i in range(1000):
    delta = np.zeros_like(weights)

    for j in range(len(inp)):
        current_inp = inp[j]
        true_prediction = true_predictions[j]

        prediction = neural_networks(current_inp, weights)

        delta += (
            (prediction - true_prediction)
            * current_inp
            * learning_rate
        )

    weights = weights - delta / len(inp)

    predictions = neural_networks(inp, weights)
    error = get_error(true_predictions, predictions)

    print("Итерация:", i + 1)
    print("RMSE:", error)
    print("Weights:", weights)
    print("-------------------")




# Проверка на новых данных
test_1 = np.array([12, 4])
test_2 = np.array([3, -8])

print("12 + 4 =", neural_networks(test_1, weights))
print("3 + (-8) =", neural_networks(test_2, weights))

# Идеальные веса для сложения:
# weights = [1, 1]
'''

import numpy as np

# ============================================================
# 1. РЕШЕНИЕ ПРОБЛЕМЫ С РАСХОЖДЕНИЕМ
# ============================================================

'''
Изменяем learning_rate и количество итераций.


inp = np.array([1])
true_prediction = 0.8
weights = np.array([0.5])

learning_rate = 0.1

for i in range(100):
    prediction = inp.dot(weights)
    error = (true_prediction - prediction) ** 2

    delta = (prediction - true_prediction) * inp * learning_rate
    weights = weights - delta

    print("Итерация:", i)
    print("Prediction:", prediction)
    print("Error:", error)
    print("Weights:", weights)
    print("-------------------")

'''
# ============================================================
# 2. ГРАДИЕНТНЫЙ СПУСК С НЕСКОЛЬКИМИ ВХОДАМИ
# ============================================================


#Изменяем learning_rate и количество итераций.
'''

inp = np.array([1, 2, 3])
true_prediction = 0.5
weights = np.array([0.1, 0.2, 0.3])

learning_rate = 0.001

for i in range(1000):
    prediction = inp.dot(weights)
    error = (true_prediction - prediction) ** 2

    delta = (prediction - true_prediction) * inp * learning_rate
    weights = weights - delta

    print("Итерация:", i)
    print("Prediction:", prediction)
    print("Error:", error)
    print("Weights:", weights)
    print("-------------------")

'''
# ============================================================
# 3. ГРАДИЕНТНЫЙ СПУСК С НЕСКОЛЬКИМИ ВЫХОДАМИ
# ============================================================


#Изменяем inp и наблюдаем за обучением.
'''

inp = 200

true_predictions = np.array([50, 100, 150])
weights = np.array([0.2, 0.3, 0.4])

learning_rate = 0.00001

for i in range(100):
    predictions = inp * weights
    error = (true_predictions - predictions) ** 2

    delta = (predictions - true_predictions) * inp * learning_rate
    weights = weights - delta

    print("Итерация:", i)
    print("Predictions:", predictions)
    print("Errors:", error)
    print("Weights:", weights)
    print("-------------------")


# ============================================================
# 4. ГРАДИЕНТНЫЙ СПУСК С НЕСКОЛЬКИМИ ВХОДАМИ И ВЫХОДАМИ
# ============================================================

'''
#Изменяем true_predictions.
'''

inp = np.array([1, 2])

true_predictions = np.array([70, 90])

weights = np.array([
    [0.1, 0.2],
    [0.3, 0.4]
])

learning_rate = 0.01

for i in range(100):
    predictions = inp.dot(weights)
    error = (true_predictions - predictions) ** 2

    delta = np.outer(inp, predictions - true_predictions) * learning_rate
    weights = weights - delta

    print("Итерация:", i)
    print("Predictions:", predictions)
    print("Error:", error)
    print("Weights:", weights)
    print("-------------------")


# ============================================================
# 5. ОБУЧЕНИЕ НА НЕСКОЛЬКИХ НАБОРАХ ДАННЫХ
# ============================================================

'''
#Изменяем количество эпох и функцию get_error.


def neural_networks(inp, weights):
    return inp.dot(weights)



def get_error(true_value, prediction_value):
    return np.sqrt(np.mean((true_value - prediction_value) ** 2))


inp = np.array([
    [150, 40],
    [170, 80],
    [160, 90]
])

true_predictions = np.array([50, 120, 140])

weights = np.array([0.2, 0.3])

learning_rate = 0.00001

# Количество эпох изменено с 500 на 100
epochs = 100

for i in range(epochs):
    error = 0

    for j in range(len(inp)):
        current_inp = inp[j]
        true_prediction = true_predictions[j]

        prediction = neural_networks(current_inp, weights)

        error += get_error(true_prediction, prediction)

        print(
            "Prediction: %.10f, True_prediction: %.10f, Weights: %s"
            % (prediction, true_prediction, weights)
        )

        delta = ((prediction - true_prediction)* current_inp* learning_rate)

        weights = weights - delta

        print("Errors: %.10f" % error)
        print("-------------------")

    print("Эпоха:", i + 1)
    print("Общая ошибка:", error)
    print("Итоговые веса:", weights)
    print("===================")

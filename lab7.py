import numpy as np

def neuralNetwork(inps, weights):
    return inps.dot(weights)

def relu(x):
    return np.maximum(0, x)

# Если значение больше 0 — возвращаем True (1).Если меньше или равно 0 — возвращаем False (0).
def reluderif(x):
    return (x > 0)

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

def update_weights_out(weights_out, layer_hid, layer_out_delta, learning_rate):
    weights_out += learning_rate * layer_hid.T.dot(layer_out_delta)
    return weights_out

def update_weights_h(weights_h, layer_in, layer_hid_delta, learning_rate):
    weights_h += learning_rate * layer_in.T.dot(layer_hid_delta)
    return weights_h


inps = np.array([
    [15, 10],
    [15, 15],
    [15, 20],
    [25, 10]
])


true_predictions = np.array([[10, 20, 15, 20]]).T
layer_hid_size = 3  # Количество нейронов в скрытом слое.
layer_in_size = len(inps[0])  # Количество входных нейронов.
layer_out_size = len(true_predictions[0])  # Количество выходных нейронов.
learning_rate = 0.0001  # Скорость обучения.
num_epochs = 100  # Количество эпох обучения.
np.random.seed(42)


# Создаём случайные веса входного и скрытого слоёв.
weights_h = 2 * np.random.random(
    (layer_in_size, layer_hid_size)
) - 1

# Создаём случайные веса скрытого и выходного слоёв.
weights_out = 2 * np.random.random(
    (layer_hid_size, layer_out_size)
) - 1



for epoch in range(num_epochs):

    layer_out_error = 0# Обнуляем суммарную ошибку перед каждой эпохой.

    # Перебираем каждый обучающий пример.
    for inp in range(len(inps)):

        # Берём одну строку входных данных.
        layer_in = inps[inp:inp + 1]

        # ПРЯМОЙ ПРОХОД.


        layer_hid = neuralNetwork(layer_in, weights_h)    # Получаем значения скрытого слоя.
        layer_hid = relu(layer_hid)# Применяем функцию активации ReLU.
        layer_out = neuralNetwork(layer_hid, weights_out)     # Получаем предсказание нейросети.


        # РАСПРОСТРАНЕНИЕ ОШИБКИ.

        layer_out_error += np.sum(
            get_error(true_predictions[inp:inp + 1], layer_out)
        )
        # Рассчитываем ошибку выходного слоя.
        layer_out_delta = true_predictions[inp:inp + 1] - layer_out
        # Передаём ошибку обратно в скрытый слой.
        layer_hid_delta = (
            layer_out_delta.dot(weights_out.T)
            * reluderif(layer_hid)
        )


        # ОБНОВЛЕНИЕ ВЕСОВ.

        # Обновляем веса выходного слоя.
        weights_out = update_weights_out(
            weights_out,
            layer_hid,
            layer_out_delta,
            learning_rate
        )
        # Обновляем веса скрытого слоя.
        weights_h = update_weights_h(
            weights_h,
            layer_in,
            layer_hid_delta,
            learning_rate
        )


        # Выводим предсказание и правильный ответ.
        print(
            "Predictions: %s, True predictions: %s"
            % (layer_out, true_predictions[inp:inp + 1])
        )


    # Выводим результаты завершённой эпохи.
    print("Epoch:", epoch + 1)
    print("Errors: %.4f" % layer_out_error)
    print("-------------")


# После обучения выводим полученные веса.
print("Обученные веса скрытого слоя:")
print(weights_h)

print("Обученные веса выходного слоя:")
print(weights_out)

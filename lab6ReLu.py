
import numpy as np

def neural_networks(inp, weights):
    return inp.dot(weights)

def relu(x):
    return np.maximum(0, x)


# Входные данные
inp = np.array([
    [15, 10],
    [15, 15],
    [15, 20],
    [25, 10]
])

# Правильные ответы
true_prediction = np.array([[10, 20, 15, 20]]).T


# Размеры слоёв
layer_in_size = len(inp[0])
layer_hid_size = 4
layer_hid_2_size = 3
layer_out_size = len(true_prediction[0])


# Создаём веса
weights_hid = 2 * np.random.random(
    (layer_in_size, layer_hid_size)
) - 1

weights_hid_2 = 2 * np.random.random(
    (layer_hid_size, layer_hid_2_size)
) - 1

weights_out = 2 * np.random.random(
    (layer_hid_2_size, layer_out_size)
) - 1


# Первый скрытый слой
prediction_hid = relu(
    neural_networks(inp, weights_hid)
)

# Второй скрытый слой
prediction_hid_2 = relu(
    neural_networks(prediction_hid, weights_hid_2)
)

# Выходной слой
prediction = neural_networks(
    prediction_hid_2, weights_out
)


# Вывод результатов
print("Первый скрытый слой:")
print(prediction_hid)

print("Второй скрытый слой:")
print(prediction_hid_2)

print("Итоговые предсказания:")
print(prediction)

print("Правильные ответы:")
print(true_prediction)

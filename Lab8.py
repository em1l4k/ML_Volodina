import numpy as np
from tensorflow.keras.datasets import mnist

# 1. Функция активации ReLU
def relu(x):
    return x * (x > 0)

# Производная функции ReLU
def reluderiv(x):
    return (x > 0)

# 2. Задаём параметры
train_images_count = 1000
test_images_count = 10000
pixels_per_image = 28 * 28
digits_num = 10


# 3. Загружаем базу данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# 4. Подготавливаем обучающие изображения
train_images = x_train[:train_images_count].reshape(
    train_images_count, pixels_per_image
) / 255.0

train_labels = y_train[:train_images_count]


# 5. Подготавливаем тестовые изображения
test_images = x_test[:test_images_count].reshape(
    test_images_count, pixels_per_image
) / 255.0

test_labels = y_test[:test_images_count]


# 6. Преобразуем правильные ответы в One-Hot Encoding
one_hot_labels = np.zeros((len(train_labels), digits_num))

for i in range(len(train_labels)):
    one_hot_labels[i][train_labels[i]] = 1

train_labels = one_hot_labels

# То же самое делаем для тестовых ответов
one_hot_labels = np.zeros((len(test_labels), digits_num))

for i in range(len(test_labels)):
    one_hot_labels[i][test_labels[i]] = 1

test_labels = one_hot_labels


# 7. Создаём матрицы весов
np.random.seed(42)

hidden_size = 50

weight_hid = (
    0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
)

weight_out = (
    0.2 * np.random.random((hidden_size, digits_num)) - 0.1
)


# 8. Параметры обучения
learning_rate = 0.001
num_epoch = 100


# 9. Обучение нейронной сети
for epoch in range(num_epoch):

    correct_answers = 0

    for j in range(len(train_images)):

        # Входной слой
        layer_in = train_images[j:j + 1]

        # Скрытый слой
        layer_hid = relu(np.dot(layer_in, weight_hid))

        # Выходной слой
        layer_out = np.dot(layer_hid, weight_out)

        # Проверяем правильность предсказания
        correct_answers += int(
            np.argmax(layer_out) ==
            np.argmax(train_labels[j:j + 1])
        )

        # Вычисляем ошибку выходного слоя
        layer_out_delta = (
            layer_out - train_labels[j:j + 1]
        )

        # Передаём ошибку обратно на скрытый слой
        layer_hid_delta = (
            layer_out_delta.dot(weight_out.T)
            * reluderiv(layer_hid)
        )

        # Обновляем веса выходного слоя
        weight_out -= (
            learning_rate
            * layer_hid.T.dot(layer_out_delta)
        )

        # Обновляем веса скрытого слоя
        weight_hid -= (
            learning_rate
            * layer_in.T.dot(layer_hid_delta)
        )

    # Выводим точность после каждой эпохи
    accuracy = correct_answers * 100 / len(train_images)

    print(
        f"Epoch: {epoch + 1}, "
        f"Accuracy: {accuracy:.2f}%"
    )


# 10. Проверяем нейронную сеть на тестовых данных
correct_answers = 0

for j in range(len(test_images)):

    layer_in = test_images[j:j + 1]

    layer_hid = relu(np.dot(layer_in, weight_hid))

    layer_out = np.dot(layer_hid, weight_out)

    correct_answers += int(
        np.argmax(layer_out) ==
        np.argmax(test_labels[j:j + 1])
    )


# Итоговая точность
test_accuracy = (
    correct_answers * 100 / len(test_images)
)

print("--------------------------------")
print(f"Test Accuracy: {test_accuracy:.2f}%")

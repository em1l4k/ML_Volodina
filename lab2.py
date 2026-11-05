import numpy as np

''''#Библиотека NumPy

#Создайте одномерный массива NumPy из 5 элементов.
arr1=np.array([1,2,3,4,5])
#print(arr1)

#Умножьте созданный массив на 2 и выведите результат на экран.
arr1=arr1*2
#print(arr1)

#Создайте два двумерных массива NumPy размером (3, 3) со случайными значениями.
arr2=np.random.rand(3,3)
arr3=np.random.rand(3,3)
print(arr2)
print(arr3)
#Перемножьте два созданных двумерных массива и выведите результат на экран.
print(arr2*arr3)

#Создайте одномерный массив NumPy из 10 случайных целых чисел. Примечание: для этого Вам понадобится изучить функцию np.random.randint().
#arr4=np.random.randint(1,10,10)
arr4=[2, 6, 2, 4, 9, 7, 7, 9, 2, 8]
print(arr4)
#Используйте индексацию и срезы, чтобы вывести только чётные элементы из массива, созданного в предыдущем задании.
print(arr4[0:4],arr4[-2:])
#Выведите среднее значение, стандартное отклонение, максимальное и минимальное значение для созданного ранее массива из 10 случайных целых чисел.
print('Среднее знач', np.mean(arr4))
print('стандартное отклонение',np.std(arr4))
print('Макс',np.max(arr4))
print('Мин',np.min(arr4))

#Упрощение кода с помощью NumPy
#Напишите по памяти код из урока "Упрощение кода с помощью NumPy".
def Neutal_Network(inp,weights):
    prediction_0=inp.dot(weights[0])
    prediction_1=prediction_0.dot(weights[1])
    return prediction_1
inp=np.array([23,45])

weight_0_1 = [0.4, 0.1]
weight_0_2 = [0.3, 0.2]

weight_1_1 = [0.4, 0.1]
weight_1_2 = [0.3, 0.1]

weight_0=np.array([weight_0_1,weight_0_2]).T
weight_1=np.array([weight_1_1,weight_1_2]).T

weights=[weight_0,weight_1]
print(Neutal_Network(inp,weights))'''
#Добавьте еще один скрытый слой к нейросети. Создайте третий набор весов weights_h_3 = [0.6, 0.2] и weights_out_3 = [0.7, 0.4]. Добавьте его в переменные weights_h и weights_out. Измените функцию neural_network так, чтобы она работала с этим третьим слоем.
def Neutal_Network(inp,weights):
    prediction_1=inp.dot(weights[0])
    prediction_2=prediction_1.dot(weights[1])
    prediction_3 = prediction_2.dot(weights[1])
    return prediction_3
inp=np.array([23,45])

weight_1_1 = [0.4, 0.1]
weight_1_2 = [0.3, 0.2]

weight_2_1 = [0.4, 0.1]
weight_2_2 = [0.3, 0.1]

weights_h_3 = [0.6, 0.2]
weights_out_3 = [0.7, 0.4]

#weight_0=np.array([weight_1_1,weight_1_2]).T
#weight_1=np.array([weight_2_1,weight_2_2]).T
#weight_3=np.array([weights_h_3,weights_out_3]).T


#Замените списки с фиксированными весами на массивы numpy, причём с генерацией случайных значений в них.
weight_1=np.random.rand(1,3)
weight_2=np.random.rand(1,3)
weight_3=np.random.rand(1,3)
print(weight_1,weight_2)
weights=weight_1,weight_2,weight_3

print(Neutal_Network(inp,weights))


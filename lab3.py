import numpy as np
#Оценка ошибки
def neural_networks(inp, weights):
    return inp.dot(weights)

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


weights=[0.2, 0.3]
imp=np.array([150,40])
prediction = neural_networks(imp,weights )
true_prediction = 50

while get_error(true_prediction,prediction)>0.001:
    weights[0]+=0.001
    weights[1] += 0.001
    prediction = neural_networks(imp,weights )


print(weights,prediction)


'''
#реализуем простейшую нейронную сеть с одним входным и выходным нейронами
def neural_networks(inp, weight):
    return inp * weight

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

inp = 0.9
weight = 0.2

true_prediction = 0.2
print(get_error(true_prediction, neural_networks(inp, weight)))

#Обучим нашу простейшую нейронную сеть с помощью алгоритма градиентного спуска
for i in range(10): #обучать будем на протяжении 10 итераций (эпох)
    prediction = neural_networks(inp, weight)
    error = get_error(true_prediction, prediction)
    print("Prediction: %.10f, Weight: %.5f, Error: %.20f" %(prediction, weight, error))
    delta = (prediction - true_prediction) * inp #это производная, она же градиент
    weight = weight - delta'''
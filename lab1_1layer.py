'''def NeuralNetwork(inp,weight,bias):
    predicition= inp*weight+bias
    return predicition

inputs = [150, 160, 170, 180, 190]
for i in inputs:
    print(NeuralNetwork(i,0.2,1))


def Neural_2(inps,weights):
    predict=0
    for i in range(len(weights)):
        predict+= inps[i]*weights[i]
        a.append(inps[i]*weights[i])

    return predict
a=[]
print(a,Neural_2([2,4],[0.1,0.2]))

def Neural_3(inp, weights):
    predict = [inp * w for w in weights]

    while predict[0] <= 0.5 or predict[1] <= 0.5:
        for i in range(len(weights)):
            if predict[i] <= 0.5:
                weights[i] += 0.1
                predict[i] = inp * weights[i]

    return weights


weights = Neural_3(2, [0.1, 0.2])
print("Получившиеся веса:", weights)


def Neural_4(inps, weights):
    predict = []

    for neuron_weights in weights:
        result = 0

        for i in range(len(inps)):
            result += inps[i] * neuron_weights[i]

        predict.append(result)

    return predict


weights_1=[0.2, 0.1, 0.5]
weights_2=[0.3, 0.1, 0.4]
weights_3=[0.5, 0.3, 0.2]
weights_4=[0.4, 0.2, 0.1]

ws = [weights_1,weights_2,weights_3,weights_4]
result=Neural_4([50, 165,20], ws)
print(result)
for i in range(1,len(ws)):
    while result[0]>result[i]:
        ws[i][0]-=0.1
        result = Neural_4([50, 165, 20], ws)
    while result[0]<result[i]:
        ws[i][0]+=0.1
        result = Neural_4([50, 165, 20], ws)
        print(result)
print(result)


def Neural_4(inps, weights):
    predict = []

    for neuron_weights in weights:
        result = 0

        for i in range(len(inps)):
            result += inps[i] * neuron_weights[i]

        predict.append(round(result, 2))

    return predict


weights_1 = [0.2, 0.1, 0.5]
weights_2 = [0.3, 0.1, 0.4]
weights_3 = [0.5, 0.3, 0.2]
weights_4 = [0.4, 0.2, 0.1]

ws = [weights_1, weights_2, weights_3, weights_4]
inps = [50, 165, 20]

# Подгоняем выходы остальных нейронов под первый
for i in range(1, len(ws)):
    results = Neural_4(inps, ws)

    difference = results[0] - results[i]
    ws[i][2] = round(ws[i][2] + difference / inps[2], 2)

print("Получившиеся веса:")
for w in ws:
    print(w)

print("Выходные значения:")
print(Neural_4(inps, ws))'''

def Neural_5(inps, weights):

    for j in range(len(ws)):
        predict = []

        for neuron_weights in weights[j]:
            result = 0

            for i in range(len(inps)):
                result += inps[i] * neuron_weights[i]

            predict.append(round(result, 2))
        inps=predict
    return predict


weights_1_1 = [0.4, 0.1]
weights_1_2 = [0.3, 0.2]

weights_2_1 = [0.4, 0.1]
weights_2_2 = [0.3, 0.1]


ws = [[weights_1_1,weights_1_2],[weights_2_1,weights_2_2]]
inps = [23, 45]
print(Neural_5(inps,ws))

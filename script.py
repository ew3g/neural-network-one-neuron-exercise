import numpy as np

inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
outputs = np.array([0,1,1,1])
weights = np.array([0.0, 0.0])
lerning_rate = 0.1

#activation funciton
def stepFunction(sum):
    print('step function for value: ' + str(sum))
    if(sum >= 1):
        return 1
    return 0

#função do neurônio
def calculateOutput(data):
    print('calculating output for data: ' + str(data))
    #multiply the "data" matrix and the "weights" matrix and sum the results
    sum = data.dot(weights)
    print('result of the sum and multiplication of the data and weights: ' + str(sum))
    return stepFunction(sum)

#neuron training
def train():
    total_errors = None
    
    #while the total erros is not 0(zero)
    while(total_errors != 0):
        total_errors = 0

        #fix the weight in each loop
        for i in range(len(outputs)):
            #classify the output
            calculated_output = calculateOutput(np.asarray(inputs[i]))
            print('calculated output ' + str(calculated_output))
            print('expected_output ' + str(outputs[i]))
            #calculates the classification error
            error = outputs[i] - calculated_output            
            total_errors += error
            print('difference/error between calculated output and expected output ' + str(error))
            print('total difference/total errors between calculated output and expected output ' + str(total_errors))
            

            #for each one of the wieghts, update its value with the error
            for j in range(len(weights)):
                weights[j] = weights[j] + (lerning_rate * inputs[i][j] * error)
                print('Updated weight: ' + str(weights[j]))
        
        print('Total errors: ' + str(total_errors))

print('Training neural network ')
train()
print('Neural network training completed ')
print(calculateOutput(inputs[0]))
print(calculateOutput(inputs[1]))
print(calculateOutput(inputs[2]))
print(calculateOutput(inputs[3]))
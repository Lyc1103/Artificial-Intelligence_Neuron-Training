print("\n******************************************************")
print("**************  1 Neuron - Sigmoid  ******************")
print("******************************************************")
# 參考老師講義
import sys
import math
import statistics as sta
import numpy as np
np.random.seed(1)

power = 1
input_num = 10

def meanSquareError( train_out, target ):
    return 0.5 * (target - train_out)**2

def main(test_times):
    train_in_volume = np.zeros((input_num, 3))
    train_sol_volume = np.zeros((input_num, 1))
    for i in range(input_num):
        radius = np.random.random()
        height = np.random.random()
        train_in_volume[i][0] = radius
        train_in_volume[i][1] = height
        train_in_volume[i][2] = math.pi * radius**2 * height * (-1)**i * 0.1
        train_sol_volume[i][0] = ((-1)**i+1)/2

    nn_weight = 2 * np.random.random((3, 1)) -1
    initial_weight = np.zeros((3, 1))
    for i in range(3):
        initial_weight[i][0] = nn_weight[i][0]

    want_to_know = np.zeros((1, 3))
    want_to_know[0][0] = np.random.random()
    want_to_know[0][1] = np.random.random()
    want_to_know[0][2] = math.pi * 0.5**3 * (-1)**(power+1) * 0.1


    for i in range(test_times):
        # Calculate trainOutVolume by sygmoid module
        train_out_volume = 1 / (1 + 1/np.exp(np.dot(train_in_volume, nn_weight)))

        # print("\ni = ", i)
        # print("nn_weight = ")
        # print(nn_weight)
        # print("train_out_volume = ")
        # print(train_out_volume)
        # print("train_target_volume = ")
        # print(train_sol_volume)

        # refine nn_weight
        nn_weight += np.dot(train_in_volume.T, (train_sol_volume -
                                                train_out_volume)*train_out_volume*(1-train_out_volume))
    else:
        print("\ntrain_in_volume = ")
        print(train_in_volume)
        print("want_to_know = ")
        print(want_to_know.T)

        print("\nintial weight = ")
        print(initial_weight)
        print("final weight = ")
        print(nn_weight)

        mean_square_error = meanSquareError(train_out_volume, train_sol_volume)
        sum = 0
        for i in range(len(mean_square_error)):
            sum += mean_square_error[i][0]
        print("Mean Square Error of training data is ", sum/input_num)

        prediction = float(1 /(1 + 1/np.exp(np.dot(want_to_know, nn_weight))))
        print("\nThe final prediction is ", prediction)
        print("The correct answer is  ", power)
        print("Mean Square Error of prediction is ", meanSquareError(prediction, power))
    print("=============== END OF TRAINING ===============\n")
# main function end

while(True):
    print("\n=============== START  TRAINING ===============")
    if(len(sys.argv) == 3):
        main( int(sys.argv[1]) )
        print("\nProcess is terminated...")
        break
    else:
        while(True):
            string = input("How many times do you want to test the neuron ?\n( press 0 to quit ) ")
            if(str.isdigit(string)):
                test_times = int(string)
                break
            else:
                print("\nInput Error ! Please enter again...")

    if(test_times == 0):
        print("\nProcess is terminated...")
        break
    else:
        main(test_times)
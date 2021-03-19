print("\n******************************************************")
print("**************  2 Neurons -  Tanh  *******************")
print("******************************************************")
# 參考老師講義
# 參考 Wikipedia https://en.wikipedia.org/wiki/Activation_function
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import sys
import math
import numpy as np
np.random.seed(1)

data_range = 1
power = 1
input_num = 10

def meanSquareError( train_out, target ):
    return 0.5 * (target - train_out)**2

def dydx(m):
    return (1+m)*(1-m)

def tanh(m1, m2):
    e_x = np.exp(np.dot(m1, m2))
    return (e_x - 1/e_x) / (e_x + 1/e_x)
def main(test_times):
    train_in_volume = np.zeros((10, 3))
    train_sol_volume = np.zeros((10, 1))
    for i in range(input_num):
        radius = data_range * np.random.random()
        height = data_range * np.random.random()
        train_in_volume[i][0] = radius
        train_in_volume[i][1] = height
        train_in_volume[i][2] = math.pi * radius**2 * height * (-1)**i
        train_sol_volume[i][0] = ((-1)**i+1)/2

    nn_weight_1 = 2 * np.random.random((3,1)) - 1
    nn_weight_2 = 2 * np.random.random((1,1)) - 1
    initial_weight_1 = np.zeros((3, 1))
    initial_weight_2 = np.zeros((1, 1))
    for i in range(3):
        initial_weight_1[i][0] = nn_weight_1[i][0]
    initial_weight_2[0][0] = nn_weight_2[0][0]

    want_to_know = np.zeros((1, 3))
    want_to_know[0][0] = np.random.random()
    want_to_know[0][1] = np.random.random()
    want_to_know[0][2] = math.pi * 0.5**3 * (-1)**(power+1)

    for i in range(test_times):
        # Calculate trainOutVolume by sygmoid module
        train_out_volume_1 = np.dot(train_in_volume, nn_weight_1) * 0.1
        train_out_volume_2 = tanh(train_out_volume_1, nn_weight_2)
        train_out_volume = train_out_volume_2

        # print("\ni = ", i)
        # print("train_out_volume 1 = ")
        # print(train_out_volume_1)
        # print("train_out_volume = ")
        # print(train_out_volume)
        # print("train_target_volume = ")
        # print(train_sol_volume)

        # refine nn_weight
        nn_weight_2 += np.dot(train_out_volume_1.T,
                                    (train_sol_volume -train_out_volume) * dydx(train_out_volume))
        train_out_previous = np.dot( train_out_volume, np.linalg.inv(nn_weight_2))
        nn_weight_1 += np.dot(train_in_volume.T,
                                    (train_sol_volume - train_out_previous) * dydx(train_out_previous) )
    else:
        print("\ntrain_in_volume = ")
        print(train_in_volume)
        print("want_to_know = ")
        print(want_to_know.T)

        print("\nintial weight 1 = ")
        print(initial_weight_1)
        print("intial weight 2 = ")
        print(initial_weight_2)
        # print("int( np.dot(train_in_volume, initial_weight_1), initial_weight_2))

        print("\nfinal weight 1 = ")
        print(nn_weight_1)
        print("final weight 2 = ")
        print(nn_weight_2)
        # print("final output = ")
        # print(train_out_volume)

        mean_square_error = meanSquareError(train_out_volume, train_sol_volume)
        sum = 0
        for i in range(len(mean_square_error)):
            sum += mean_square_error[i][0]
        print("Mean Square Error of training data is ", sum/input_num)

        prediction = float(tanh( np.dot(want_to_know, nn_weight_1)*0.1, nn_weight_2))
        print("\nThe final prediction is ", prediction)
        print("The correct answer is  ", power)
        print("Mean Square Error is ", meanSquareError(power, prediction))
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
            string = input("How many times do you want to test these two neurons ?\n( press 0 to quit ) ")
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
print("\n******************************************************")
print("************** 3 Neurons - Sigmoid *******************")
print("******************************************************")
# 參考老師講義
# 線性訓練
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import sys
import math
import numpy as np
np.random.seed(1)

power = 1
input_num = 10
data_range = 0.1 # from 0 to data_range
sigma_range_mult = 1
sigma_range_sub = 0
bias = 0

def meanSquareError( train_out, target ):
    return 0.5 * (target - train_out)**2

def dydx(m):
    return m*(1-m)

def sigmoid(m):
    return 1 / \
        (1 + 1/np.exp(m))

def trainOut(train_in_volume, nn_weight_1, nn_weight_2, nn_weight_3):
    train_out_volume_1 = np.dot(train_in_volume, nn_weight_1) * sigma_range_mult - sigma_range_sub + bias
    train_out_volume_2 = np.dot(train_in_volume, nn_weight_2) * sigma_range_mult - sigma_range_sub + bias
    volumn_hstack = np.hstack([train_out_volume_1, train_out_volume_2])
    train_out_volume_3 = np.dot(volumn_hstack, nn_weight_3)
    train_out_volume = sigmoid(train_out_volume_3)
    return train_out_volume

def main(test_times):
    train_in_volume = np.zeros((input_num, 3))
    train_sol_volume = np.zeros((input_num, 1))
    for i in range(input_num):
        radius = data_range * np.random.random()
        height = data_range * np.random.random()
        train_in_volume[i][0] = radius
        train_in_volume[i][1] = height
        train_in_volume[i][2] = math.pi * radius**2 * height * (-1)**i *1000000
        train_sol_volume[i][0] = ((-1)**i+1)/2


    nn_weight_1 = 2 * np.random.random((3, 1)) -1
    nn_weight_2 = 2 * np.random.random((3, 1)) -1
    nn_weight_3 = 2 * np.random.random((2, 1)) -1
    initial_weight_1 = np.zeros((3,1))
    initial_weight_2 = np.zeros((3,1))
    initial_weight_3 = np.zeros((2,1))
    for i in range(3):
        initial_weight_1[i][0] = nn_weight_1[i][0]
        initial_weight_2[i][0] = nn_weight_2[i][0]
    initial_weight_3[0][0] = nn_weight_3[0][0]
    initial_weight_3[1][0] = nn_weight_3[1][0]

    want_to_know = np.zeros((1, 3))
    want_to_know[0][0] = np.random.random()
    want_to_know[0][1] = np.random.random()
    want_to_know[0][2] = math.pi * 0.5**3 * (-1)**(power+1) *1000000


    for i in range(test_times):
        # Calculate trainOutVolume by sygmoid module
        train_out_volume_1 = np.dot(train_in_volume, nn_weight_1) * sigma_range_mult - sigma_range_sub + bias
        train_out_volume_2 = np.dot(train_in_volume, nn_weight_2) * sigma_range_mult - sigma_range_sub + bias
        volumn_hstack = np.hstack([train_out_volume_1, train_out_volume_2])
        train_out_volume_3 = np.dot(volumn_hstack, nn_weight_3)
        train_out_volume = sigmoid(train_out_volume_3)

        # print("\n==================    i = ", i, "    ==================")
        # print("weight 1 = ")
        # print(nn_weight_1)
        # print("weight 2 = ")
        # print(nn_weight_2)
        # print("weight 3 = ")
        # print(nn_weight_3)
        # print("train_out_volume 1 = ")
        # print(train_out_volume_1)
        # print("train_out_volume 2 = ")
        # print(train_out_volume_2)
        # print("train_out_volume = ")
        # print(train_out_volume)
        # print("train_target_volume = ")
        # print(train_sol_volume)

        # refine nn_weight
        nn_weight_3 += np.dot(volumn_hstack.T, (train_sol_volume - train_out_volume) * train_out_volume*(1-train_out_volume))
        train_out_previous = np.dot( train_out_volume, np.linalg.pinv(nn_weight_3))
        train_out_previous_hsplit1, train_out_previous_hsplit2 = np.hsplit(train_out_previous, [1])
        nn_weight_1 += np.dot(train_in_volume.T, (train_sol_volume - train_out_previous_hsplit1) * dydx(train_out_previous_hsplit1) )
        nn_weight_2 += np.dot(train_in_volume.T, (train_sol_volume - train_out_previous_hsplit2) * dydx(train_out_previous_hsplit2) )
    else:
        print("\ntrain_in_volume = ")
        print(train_in_volume)
        print("want_to_know = ")
        print(want_to_know)
        print("\nintial weight 1 = ")
        print(initial_weight_1)
        print("intial weight 2 = ")
        print(initial_weight_2)
        print("intial weight 3 = ")
        print(initial_weight_3)
        # print("intial output = ")
        # print(trainOut(train_in_volume, initial_weight_1, initial_weight_2, initial_weight_3))

        print("\nfinal weight 1 = ")
        print(nn_weight_1)
        print("final weight 2 = ")
        print(nn_weight_2)
        print("final weight 3 = ")
        print(nn_weight_3)
        # print("final output = ")
        # print(trainOut(train_in_volume, nn_weight_1, nn_weight_2, nn_weight_3))

        mean_square_error = meanSquareError(train_out_volume, train_sol_volume)
        sum = 0
        for i in range(len(mean_square_error)):
            sum += mean_square_error[i][0]
        print("Mean Square Error of training data is ", sum/input_num)

        prediction = float(trainOut(want_to_know, nn_weight_1, nn_weight_2, nn_weight_3))
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
            string = input("How many times do you want to test these three neurons ?\n( press 0 to quit ) ")
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
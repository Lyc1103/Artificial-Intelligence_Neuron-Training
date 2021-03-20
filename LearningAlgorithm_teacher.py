print("\n******************************************************")
print("*************  Teacher's   Example  ******************")
print("******************************************************")
from numpy import *
train_in = array([[0, 2, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
train_sol = array([[0, 0, 0, 1, 1]]).T

# initialize nn_weights
random.seed(1)
nn_weights = 2 * random.random((3,1)) - 1

# train the network
run_time = 1000
for i in range(run_time):
    # if i == 999 or i == 0 :
    #     print("\n i = ", i, "nn_weight = ")
    #     print(nn_weights)

    # Calculate the outputs for each training examples
    train_out = 1 / (1 + 1/exp(dot(train_in, nn_weights)))
    # if i == 999 or i == 0:
    #     print("train_out = ")
    #     print(train_out)

    # Run the NN adjustments
    nn_weights += dot(train_in.T, (train_sol - train_out)
                            * train_out * (1 - train_out))
else:
    # unknow test input
    test_in = array([1, 0, 0])
    # print the result for our unknown test input
    print('\nThe final prediction is ', 1 /
          (1 + 1/exp(dot(test_in, nn_weights))))

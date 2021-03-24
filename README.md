# Artificial-Intelligence_Neuron-Training

> Author : Ya Chen <br>
> Date : 2021 / 3 / 13

> List : <br>
>
> > [If you have the `make` commend](https://github.com/Lyc1103/Artificial-Intelligence_Neuron-Training#if-you-have-the-make-commend-) <br> [If you <font color = "red">do not</font> have the `make` commend](https://github.com/Lyc1103/Artificial-Intelligence_Neuron-Training#if-you-do-not-have-the-make-commend-)

## Description

AlphaGo has won the King of Go, creating a new milestone in artificial intelligence. One of the techniques he used is the <b>Neural Network</b>, and we already know how to <b>"implement a neural-like network to compute mysterious functions".</b><br>
<br>
Suppose one day you go to a BetaGo company and start working in the field of <b>"Artificial Intelligence"</b>, and the boss has never worked on Go - no, he has never worked on Neural Network - but he orders you to experiment directly with <b><font color = "red">Neural Network</font></b>, to experience the progress of machine learning, and to report on it for future planning.<br>
<br>
You can choose a real application ( like volume of a cylinder, triangle area, etc. ), then set up a small Neural Network with a few neurons, initialize the weights, start training some of the training data, and after a period of time, the training is complete and can be used to predict the new data of the location. At this point, you can tell your BetaGo boss, is it really as good as the experts say it is? How to draw a conclusion?

## This report contains :

I choose the <b>Cylindrical Volume</b> application theme to implement the Neural Network.<br>
And here, I will show some results:

```
1. training a neuron by <b>Sigmoid</b> and by <b>Tanh</b>
2. training <b>two</b> neurons by <b>Sigmoid</b> and by <b>Tanh</b>
3. Training <b>three</b> neurons by Sigmoid
```

p.s. You will get the detail of my report at <br>

```
Artificial_Intelligence_Hw1_Written_Report.pdf
```

<br>

## If you have the `make` commend :

If your device supports the `make` command, this will be much easier ( because my file name is very long... ).<br>
You can type `make` in Terminal to see the output of all Python files directly.<br>
You can also type in :<br>

> <p>>>>make q2 <br>
> // Output the execution result of hw1 question 2 </p>

> <p>>>>make q3 <br>
> // Output the execution result of hw1 question 3 </p>

> <p>>>>make q5 <br>
> // Output the execution result of hw1 question 5 </p>

> <p>>>>make q6 <br>
> // Output the execution result of hw1 question 6 </p>

> <p>>>>make q8 <br>
> // Output the execution result of hw1 question 8 </p>

<br>

## If you <font color = "red">do not</font> have the `make` commend :

If your device <b><font color = "red">does not</font></b> supports the `make` command, there will be a little inconvenience ( because my file name is very long... ).<br>
You can type in :<br>

> <p>>>>python HW_1_CylinderVolume_OneNeuron_Sigmoid.p <br>
> // Output the execution result of hw1 question 2 </p>

> <p>>>>python HW_1_CylinderVolume_OneNeuron_Tanh.p  <br>
> // Output the execution result of hw1 question 3 </p>

> <p>>>>python HW_1_CylinderVolume_TwoNeurons_Sigmoid.p <br>
> // Output the execution result of hw1 question 5 </p>

> <p>>>>python HW_1_CylinderVolume_TwoNeurons_Tanh.py <br>
> // Output the execution result of hw1 question 6 </p>

> <p>>>>python HW_1_CylinderVolume_ThreeNeurons_Sigmoid.p <br>
> // Output the execution result of hw1 question 8 </p>

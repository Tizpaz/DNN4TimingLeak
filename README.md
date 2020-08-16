# Deep Neural Networks for Side Channel Detection and Quantification

DNN4SC is a tool for detecting and quantifying timing side channels. In this repository, you can find the source codes, the code for learning
deep neural networks, and two algorithms for quantifying the side
channels.

Please see the paper in RV'19: [Efficient Detection and Quantification of Timing Leaks with Neural Networks](https://arxiv.org/abs/1907.10159).

### System Requirement
python 2.7, tensorflow 1.15, mat4py, gurobi9, sklearn, numpy, pandas, scipy, pickle.

## Source Codes and Input Generation
We first randomly generate inputs for the programs under the test
(see the 'Program-Source-Code' folder for the source code).
The input generation
step provides us with a csv file that includes public features,
secret features, and the execution times.
The csv file is the input for the NN learning. Samples
of these files are shown inside 'Neural-Network-Learning' folder.

## Learning time model for sorting program via deep neural networks
Let's overview our approach to learn the time model of the sorting
program. Please see the program inside 'Program-Source-Codes' folder.
First, we cd into 'Neural-Network-Learning' folder:

```bash
cd Neural-Network-Learning/
```

Please make sure that the following values are set for 'learnNeuralModel-reducer.py': pub_layer_sizes = [200,200,200]; priv_layer_sizes = [1]; combined_layer_sizes = [200,25]; and the learning rate is 0.01.

```bash
python learnNeuralModel-reducer.py sort/sorting_time.csv 30000 0.1 50 0.0001 0 0 sort/sorting
```

where the first parameter is the input csv file, the second parameter is the number of iterations, third parameter is the ratio of testing, the fourth parameter is the batch size, the fifth parameter is the regularization hyper-parameter, the six parameter is the number of binarized neurons in the interface layer, the seventh parameter is whether to plot the prediction (size versus time), and the last parameter is the path to write the weight and biases
matrices to a mat file after training.

We can issue similar commands to learn other benchmark. For example, for the benchmark 'R\_4', we first set learning_rate = 1e-3, pub_layer_sizes = [5], priv_layer_sizes = [5], and combined_layer_sizes = [10], and then
we issue the following command:

```bash
python learnNeuralModel-reducer.py Micro-Benchmark/R_4.csv 50000 0.1 50 0.00001 2 0 Micro-Benchmark/RegEx_4_1
```

## Quantifying Side-channel with counting algorithm v.1 (RV19)
After learning neural networks, where it gives us the matrices
of weights and biases, we want to use this information to count
how many secret values map to a particular value of interface layer output
(note that the interface layer is binarized and outputs 0 and 1).
Let's go inside the counting folder:

```bash
cd Counting-V1-RV19/
```

Now, we use the matrices and build MILP formulations in
the Gurobi optimization framework. For example of the benchmark 'R\_4',
we issue the following command:

```bash
python main.py Micro-Benchmark/R_4/RegEx_4_1_output_4.mat Micro-Benchmark/R_4/RegEx_4_1_MIP.py 2
```

```bash
python main.py Micro-Benchmark/R_4/RegEx_4_1_output_4.mat Micro-Benchmark/R_4/RegEx_4_1_MIP.py 2
```
Next, we run the optimization for each possible outcome
values of interface layer. For the example of 'R\_4', we issue the following
command:

```bash
gurobi.sh Micro-Benchmark/R_4/RegEx_4_1_MIP.py
```

Finally, we run the script to measure various quantitative information flow:

```bash
python post_process_MIP.py Micro-Benchmark/R_4/RegEx_4_1_MIP.py_Model_res.txt
```

## Quantifying Side-channel with counting algorithm v.2
We also develop an efficient algorithm for counting number of secret
values in each value of interface layer. Let's go and visit this
algorithm:

```bash
cd Counting-V2/
```

The commands for the new algorithm follows the same structure as
version 1. However, the new algorithm also performs both formulations
and optimizations in one step. Let's do this for the another benchmark 'Branch\_Loop\_3':

```bash
python main.py Micro-Benchmark/Branch_Loop_3/Branch_Loop_3_output.mat Micro-Benchmark/Branch_Loop_3/Branch_Loop_3_MIP_obj.py 6
python post_process_MIP.py Micro-Benchmark/Branch_Loop_3/Branch_Loop_3_MIP_obj_MIP.py_Model_res.txt
```

Similarly, for 'R\_4' benchmark, we can issue the following commands:

```bash
python main.py Micro-Benchmark/R_4/RegEx_4_1_output_4.mat Micro-Benchmark/R_4/RegEx_4_1_MIP.py 2
python post_process_MIP.py Micro-Benchmark/R_4/RegEx_4_1_MIP_MIP.py_Model_res.txt
```

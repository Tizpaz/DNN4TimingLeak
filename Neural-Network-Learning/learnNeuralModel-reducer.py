# secret input is passing through a k-bit reducer box.
# requirements: python 2.7, tensorflow 1.15, sklearn, numpy, pandas, scipy, pickle,
import tensorflow as tf
import numpy as np
import sys
import random
import scipy.io
import pandas as pd
import csv
import matplotlib.pyplot as plt
# import information_process  as info_proc
# import information_process_tensor as info_proc_tens
from numpy.random import seed
from tensorflow import set_random_seed
from sklearn.feature_selection import mutual_info_regression
import scipy.stats as stats
import entropy_estimators as ee
import pickle
sys.path.append(tf.__path__[0]+'/python/ops')
from tensorflow.python.ops import gen_array_ops
import time
from sklearn.metrics import r2_score

def hard_sigmoid(x):
    return tf.clip_by_value((x + 1.)/2., 0, 1)

def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    a op that behave as f(x) in forward mode,
    but as g(x) in the backward mode.
    '''
    rounded = tf.round(x)
    return x + tf.stop_gradient(rounded-x)

# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1
# during back propagation
def binary_tanh_unit(x):
    return 2.*round_through(hard_sigmoid(x))-1.

def binary_sigmoid_unit(x):
    return round_through(hard_sigmoid(x))

class LearningData:

    def __init__(self, csv_filename, num_pub_inputs, num_priv_inputs, test_set_size):
        # Load the data from the csv_filename
        self.__all_rows = []
        self.__test_size = test_set_size
        self.__num_inputs = num_pub_inputs + num_priv_inputs
        self.__num_pub_inputs = num_pub_inputs
        self.__num_priv_inputs = num_priv_inputs

        with open(csv_filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for line in csvreader:
                row = []
                for x in line:
                    row.append(float(x))
                self.__all_rows.append(row)
            csvfile.close()
        n_all = len(self.__all_rows)
        np.random.shuffle(self.__all_rows)

    def get_num_inputs(self):
        return self.__num_inputs

    def get_num_inputs(self):
        return self.__num_inputs

    def get_num_all_rows(self):
        return len(self.__all_rows)

    def read_whole_file(self):
        file = open(csv_filename,'r')
        data = []
        all_data = file.readlines()
        file.close()
        X_all = np.zeros((len(all_data)-1,len(all_data[0].split(','))-1))
        Y_all = np.zeros((len(all_data)-1,1))
        for i,line in enumerate(all_data):
            if i==0:
                continue
            tokens = line.split(',')
            p0,s0,time = tokens[0:self.__num_pub_inputs],tokens[self.__num_pub_inputs:-1],tokens[-1]
            p0 = np.array(p0)
            s0 = np.array(s0)
            p0,s0,time = p0.astype(np.float32),s0.astype(np.float32),float(time)
            l = p0.tolist()
            l.extend(s0.tolist())
            X_all[i-1]=l
            Y_all[i-1]=time/1000
        return X_all,Y_all

    def get_random_inputs_and_outputs(self, batch_size):
        num_inputs = self.__num_inputs
        num_pub_inputs = self.__num_pub_inputs
        num_priv_inputs = self.__num_priv_inputs
        n = len(self.__all_rows) - self.__test_size
        # choose batch_size random indices from 0 to n-1
        lst = random.sample(range(n), batch_size)
        pub_x_values = np.zeros(shape=(batch_size,num_pub_inputs))
        priv_x_values = np.zeros(shape=(batch_size, num_priv_inputs))
        y_values = np.zeros(shape=(batch_size,1))
        count = 0
        for i in lst:
            row_lst = self.__all_rows[i]
            pub_x_values[count, :] = np.matrix(row_lst[0:num_pub_inputs]).reshape(1, num_pub_inputs)
            priv_x_values[count, :] = np.matrix(row_lst[num_pub_inputs:num_inputs]).reshape(1, num_priv_inputs)
            y_values[count,0] = np.matrix([row_lst[num_inputs]])/1000
            count = count + 1
        return pub_x_values, priv_x_values, y_values

    def get_inputs_and_outputs(self, index, batch_size):
        num_inputs = self.__num_inputs
        num_pub_inputs = self.__num_pub_inputs
        num_priv_inputs = self.__num_priv_inputs
        n = len(self.__all_rows)
        # choose batch_size random indices from 0 to n-1
        all_values = np.zeros(shape=(batch_size,num_pub_inputs + num_priv_inputs))
        pub_x_values = np.zeros(shape=(batch_size,num_pub_inputs))
        priv_x_values = np.zeros(shape=(batch_size, num_priv_inputs))
        y_values = np.zeros(shape=(batch_size,1))
        start = index * batch_size
        end = ((index+1) * batch_size) - 1
        count = 0
        while start < end and start < n:
            row_lst = self.__all_rows[start]
            all_values[count, :] = np.matrix(row_lst[0:num_inputs]).reshape(1, num_pub_inputs + num_priv_inputs)
            pub_x_values[count, :] = np.matrix(row_lst[0:num_pub_inputs]).reshape(1, num_pub_inputs)
            priv_x_values[count, :] = np.matrix(row_lst[num_pub_inputs:num_inputs]).reshape(1, num_priv_inputs)
            y_values[count,0] = np.matrix([row_lst[num_inputs]])/1000
            start = start + 1
            count = count + 1
        return pub_x_values, priv_x_values, all_values, y_values

    def get_test_set(self):
        num_inputs = self.__num_inputs
        n = len(self.__all_rows)
        count = 0
        batch_size = self.__test_size
        pub_x_values = np.zeros(shape=(batch_size, num_pub_inputs))
        priv_x_values = np.zeros(shape=(batch_size, num_priv_inputs))
        y_values = np.zeros(shape=(batch_size,1))
        count = 0
        for i in range(n - self.__test_size, n):
            row_lst = self.__all_rows[i]
            pub_x_values[count, :] = np.matrix(row_lst[0:num_pub_inputs]).reshape(1, num_pub_inputs)
            priv_x_values[count, :] = np.matrix(row_lst[num_pub_inputs:num_inputs]).reshape(1, num_priv_inputs)
            y_values[count,0] = np.matrix([row_lst[num_inputs]])/1000
            count = count + 1
        return pub_x_values, priv_x_values, y_values

def exctract_activity(sess, fn, num_batch, batch_size, hidden_pub, hidden_pri, hidden_joint, y_mdl_var, pub_x, priv_x, y):
    w_temp_pub = []
    w_temp_pri = []
    w_temp_joint = []
    y_mdl = np.zeros(shape=(fn.get_num_all_rows(),1))
    cnt = -1
    for i in range(0,(int(num_batch))):
        batch_X_pub, batch_X_priv, _, batch_Y = fn.get_inputs_and_outputs(i, batch_size)
        w_temp_local_pub = sess.run([hidden_pub], feed_dict={pub_x: batch_X_pub, priv_x: batch_X_priv, y: batch_Y})
        w_temp_local_pri = sess.run([hidden_pri], feed_dict={pub_x: batch_X_pub, priv_x: batch_X_priv, y: batch_Y})
        w_temp_local_joint = sess.run([hidden_joint], feed_dict={pub_x: batch_X_pub, priv_x: batch_X_priv, y: batch_Y})
        y_mdl_eval = sess.run(y_mdl_var, feed_dict={pub_x: batch_X_pub, priv_x: batch_X_priv, y: batch_Y})
        for e in y_mdl_eval:
            cnt += 1
            y_mdl[cnt] = e

        for s in range(len(w_temp_local_pub[0])):
            if i == 0:
                w_temp_pub.append(w_temp_local_pub[0][s])
            else:
                w_temp_pub[s] = np.concatenate((w_temp_pub[s], w_temp_local_pub[0][s]), axis=0)
        for s in range(len(w_temp_local_pri[0])):
            if i == 0:
                w_temp_pri.append(w_temp_local_pri[0][s])
            else:
                w_temp_pri[s] = np.concatenate((w_temp_pri[s], w_temp_local_pri[0][s]), axis=0)

        for s in range(len(w_temp_local_joint[0])):
            if i == 0:
                w_temp_joint.append(w_temp_local_joint[0][s])
            else:
                w_temp_joint[s] = np.concatenate((w_temp_joint[s], w_temp_local_joint[0][s]), axis=0)

    return w_temp_pub, w_temp_pri, w_temp_joint, y_mdl


def setup_multilayer_network(num_inputs, inp_vars, num_outputs, layer_sizes, make_output_layer, secret_layer, out_secret):
    n0 = num_inputs
    prev_layer_in = inp_vars
    weights_list = []
    hidden = []

    if secret_layer:
        for n in layer_sizes:
            W1 = tf.Variable(tf.random_normal(shape=(n0, n), mean=0.0, stddev=1.0, dtype=tf.float32))
            b1 = tf.Variable(tf.random_normal(shape=(1, n), mean=0.1, stddev=0.5, dtype=tf.float32))
            layer1_in = tf.matmul(prev_layer_in, W1) + b1
            # layer1_out = binary_sigmoid_unit(layer1_in)
            layer1_out = tf.nn.softsign(layer1_in)
            layer1_out = gen_array_ops.quantize_and_dequantize_v2(layer1_out, 0.0, 1.0, num_bits=1, range_given=True,signed_input=False)
            weights_list.append((W1, b1))
            prev_layer_in = layer1_out
            hidden.append(prev_layer_in)
            n0 = n
    else:
        for n in layer_sizes:
            W1 = tf.Variable(tf.random_normal(shape=(n0, n), mean=0.0, stddev=1.0, dtype=tf.float32))
            b1 = tf.Variable(tf.random_normal(shape=(1, n), mean=0.1, stddev=0.5, dtype=tf.float32))
            layer1_in = tf.matmul(prev_layer_in, W1) + b1
            layer1_out = tf.nn.relu(layer1_in)
            weights_list.append((W1, b1))
            prev_layer_in = layer1_out
            hidden.append(prev_layer_in)
            n0 = n

    if secret_layer:
        W1 = tf.Variable(tf.random_normal(shape=(n0, out_secret), mean=0.0, stddev=1.0, dtype=tf.float32))
        b1 = tf.Variable(tf.random_normal(shape=(1, out_secret), mean=0.1, stddev=0.5, dtype=tf.float32))
        layer1_in = tf.matmul(prev_layer_in, W1) + b1
        # method 1
        # layer1_out = tf.divide(tf.add(tf.sign(layer1_in),1),2)
        # method 2
        # v1 = tf.Variable(tf.constant(10000,dtype=tf.float32), trainable=False)
        # v2 = tf.Variable(tf.constant(0.001,dtype=tf.float32), trainable=False)
        # layer1_out = tf.sigmoid(tf.multiply(tf.subtract(layer1_in,v2),v1))
        # method 3
        layer1_out = tf.nn.softsign(layer1_in)
        layer1_out = gen_array_ops.quantize_and_dequantize_v2(layer1_out, 0.0, 1.0, num_bits=1, range_given=True,signed_input=False)
        # method 4
        # layer1_out = binary_sigmoid_unit(layer1_in)

        weights_list.append((W1, b1))
        prev_layer_in = layer1_out
        hidden.append(prev_layer_in)
        n0 = n

    if make_output_layer:
        W3 = tf.Variable(tf.random_normal(shape=(n0, num_outputs), mean=0.0, stddev=1.0, dtype=tf.float32))
        b3 = tf.Variable(tf.random_normal(shape=(1, num_outputs), mean=0.1, stddev=0.5, dtype=tf.float32))
        y_mdl = tf.matmul(prev_layer_in, W3) + b3
        weights_list.append((W3, b3))
        hidden.append(y_mdl)
    else:
        y_mdl = prev_layer_in
    return y_mdl, weights_list, hidden

def setup_public_private_nn(fn, pub_layer_sizes, priv_layer_sizes, combined_layer_sizes,
                            num_pub_inputs, num_priv_inputs, num_iteration, batch_size,
                            learning_rate, regularizer_param, plot, output_fstem, out_secret):
    tf.set_random_seed(245)
    # First setup the input place holders
    pub_x = tf.placeholder(shape=(None, num_pub_inputs), dtype=tf.float32)
    priv_x = tf.placeholder(shape=(None, num_priv_inputs), dtype=tf.float32)
    y = tf.placeholder(shape=(None,1), dtype=tf.float32)
    public_out, weights_list_public, hidden_pub = setup_multilayer_network(num_pub_inputs, pub_x, 0, pub_layer_sizes, False, False, out_secret)
    private_out, weights_list_private, hidden_pri = setup_multilayer_network(num_priv_inputs, priv_x, 0, priv_layer_sizes, False, True, out_secret)
    # Now make a combined layer with public and private outpts
    joint_layer_in = tf.concat([public_out, private_out], 1)
    joint_layer_size = joint_layer_in.get_shape().as_list()
    # print('joint_layer_size', joint_layer_size[1])
    y_mdl, weights_list_joint, hidden_joint = setup_multilayer_network(joint_layer_size[1], joint_layer_in, 1, combined_layer_sizes, True, False, out_secret)
    regularizer = 0
    # for (W, b) in weights_list_private:
    #     regularizer += tf.nn.l2_loss(W)
    #     regularizer += tf.nn.l2_loss(b)

    # loss function is l2 norm
    y_diff = tf.square(y_mdl - y)
    # loss function is a combination of l1 and l2 losses
    # y_diff = y_mdl - y
    # comparison_gt = tf.greater(y_diff, tf.constant(0, dtype=tf.float32))
    # comparison_1 = tf.less(tf.abs(y_diff), tf.constant(1, dtype=tf.float32))
    # y_diff = tf.where(comparison_gt, tf.abs(y_diff), tf.square(y_diff))
    loss_pred = tf.reduce_mean(y_diff)
    loss_regularization = tf.reduce_mean(regularizer_param * regularizer)
    loss_fn = loss_pred + loss_regularization
    # Setting up the training
    opt = tf.train.AdamOptimizer(learning_rate)
    optimizer = opt.minimize(loss_fn)

    X_all = []
    Y_all = []
    X_all, Y_all = fn.read_whole_file()

    #if num_pub_inputs == 1:
    #    plt.scatter(X_all[:,0].reshape(-1,1),Y_all)
    #    plt.show()

    ws_pub = []
    ws_pri = []
    ws_joint = []
    startTime = int(round(time.time() * 1000))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cost_itr_tot = 0
        cost_itr_pred = 0
        cost_itr_reg = 0
        for i in xrange(num_iteration):
            batch_X_pub, batch_X_priv, batch_Y = fn.get_random_inputs_and_outputs(int(batch_size))
            _ , temp_cost, cost_pred, cost_reg = sess.run([optimizer, loss_fn, loss_pred, loss_regularization], feed_dict={pub_x: batch_X_pub, priv_x: batch_X_priv, y: batch_Y})
            cost_itr_tot += temp_cost
            cost_itr_pred += cost_pred
            cost_itr_reg += cost_reg
            if (i+1) % 1000 == 0:
                cost_itr_tot /= 1000
                cost_itr_pred /= 1000
                cost_itr_reg /= 1000
                print("Epoch:", '%04d' % (i+1), "cost_total={:.9f}".format(cost_itr_tot), "cost_pred={:.9f}".format(cost_itr_pred), "cost_reg={:.9f}".format(cost_itr_reg))
                # cost_itr_tot = 0
                # cost_itr_pred = 0
                # cost_itr_reg = 0
		# if cost_itr_pred < 300000.0:
		# 	break
            #if i % 5000 == 0 and num_pub_inputs == 1 and plot == 1:
            #    y_pred_np = sess.run(y_mdl,{pub_x:X_all[:,0].reshape(-1,num_pub_inputs),priv_x:X_all[:,1:].reshape(-1,num_priv_inputs),y:Y_all.reshape(-1,1)})
            #    plt.scatter(X_all[:,0].reshape(-1,1),y_pred_np)
            #    plt.show()

        print("Optimization Finished!")
        endTime = int(round(time.time() * 1000))
        rTime = endTime - startTime
        print('Time of computation ' + ': ' + str(rTime))
        md = {}
        i = 1
        for (W,b) in weights_list_public:
            md['W_pub_enc_%d'%(i)] = W.eval(sess)
            md['b_pub_enc_%d'%(i)] = b.eval(sess)
            i = i + 1
        i = 1
        for (W, b) in weights_list_private:
            md['W_priv_enc_%d'%(i)] = W.eval(sess)
            md['b_priv_enc_%d'%(i)] = b.eval(sess)
            i = i + 1
        i = 1
        for (W, b) in weights_list_joint:
            md['W_joint_%d'%(i)] = W.eval(sess)
            md['b_joint_%d'%(i)] = b.eval(sess)
            i = i + 1
        scipy.io.savemat('%s.mat' % output_fstem, md)
        print('Results saved to file: %s' % output_fstem)

        final_test_X_pub, final_test_X_priv , final_test_Y = fn.get_test_set()
        final_loss = loss_fn.eval(feed_dict={pub_x: final_test_X_pub, priv_x: final_test_X_priv, y: final_test_Y})
        print('Test Loss:', final_loss)
        y_pred_np = sess.run(y_mdl,{pub_x:X_all[:,0:num_pub_inputs].reshape(-1,num_pub_inputs),priv_x:X_all[:,num_pub_inputs:].reshape(-1,num_priv_inputs),y:Y_all.reshape(-1,1)})
        r2 = r2_score(Y_all.reshape(-1,1), y_pred_np)
        print('Coefficient of Determination R^2:', r2)
	    # with open('outfile_0', 'wb') as fp:
        #     pickle.dump(Y_all.reshape(1,-1), fp)
        # with open('outfile_2', 'wb') as fp:
        #     pickle.dump(y_pred_np.reshape(1,-1), fp)
        if num_pub_inputs == 1:
            plt.scatter(X_all[:,0].reshape(-1,1),y_pred_np)
            plt.savefig("scatter.png",dpi=300)
        #    plt.show()
        ones = []
        for r in range(len(y_pred_np)):
            ones.append(r+1)
        plt.scatter(ones,y_pred_np)
        plt.savefig("scatter1.png",dpi=300)

if __name__ == '__main__':
    # Example of run:
    # python learnNeuralModel.py Micro-Benchmark/Reg_Ex_4_ver.1/RegEx_4_1.csv 50000 0.1 512 0.5 1 Micro-Benchmark/Reg_Ex_4_ver.1/RegEx_4_1_output
    # recommended values for [numbder of iterations] = 100000
    # [batch_size] = 500
    if len(sys.argv) < 8:
        print('Usage: ', sys.argv[0], '[name of csv] [numbder of iterations] [ratio of test data] [batch_size] [Regularization_Rate] [plot 0 | 1] [outputfilestem]')
        sys.exit(2)

    seed(12)
    set_random_seed(245)

    csv_filename = sys.argv[1]
    df = pd.read_csv(csv_filename)
    num_rows = df.shape[0]

    learning_rate = 1e-2

    # public variables start with 'p' and secret variables start with 's'.
    # public features should appear before secret variables.
    size_public = 0
    size_secret = 0
    for column in df:
        if column.startswith('p'):
            size_public = size_public + 1
        elif column.startswith('s'):
            size_secret = size_secret + 1

    num_pub_inputs = size_public
    num_priv_inputs = size_secret


    num_iteration = int(sys.argv[2])
    ratio_test = float(sys.argv[3])
    batch_size = int(sys.argv[4])
    regularizer_param = float(sys.argv[5])
    num_secret_out = int(sys.argv[6])
    plot = int(sys.argv[7])
    output_fstem = sys.argv[8]

    pub_layer_sizes = [200,200,200]
    priv_layer_sizes = [1]
    combined_layer_sizes = [200,25]


    fn = LearningData(csv_filename, num_pub_inputs, num_priv_inputs, int(num_rows*ratio_test))
    setup_public_private_nn(fn, pub_layer_sizes, priv_layer_sizes,
                            combined_layer_sizes, num_pub_inputs,
                            num_priv_inputs, num_iteration, batch_size, learning_rate,
                            regularizer_param, plot, output_fstem,num_secret_out)

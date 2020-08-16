'''
Calculate the information in the network
Can be by the full distribution rule (for small netowrk) or bt diffrenet approximation method
'''
import multiprocessing
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
NUM_CORES = multiprocessing.cpu_count()
from mutual_information_calculation import *
import entropy_estimators as ee

def calc_information_for_layer(data, bins, unique_inverse_x, unique_inverse_y, pxs, pys1):
	bins = bins.astype(np.float32)
	digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
	b2 = np.ascontiguousarray(digitized).view(
		np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
	unique_array, unique_inverse_t, unique_counts = \
		np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
	p_ts = unique_counts / float(sum(unique_counts))
	PXs, PYs = np.asarray(pxs).T, np.asarray(pys1).T
	local_IXT, local_ITY = calc_information_from_mat(PXs, PYs, p_ts, digitized, unique_inverse_x, unique_inverse_y,
	                                                 unique_array)
	return local_IXT, local_ITY


def calc_information_sampling(data, bins, pys1, pxs, label, b, b1, len_unique_a, p_YgX, unique_inverse_x,
                              unique_inverse_y, calc_DKL=False):

	ins = bins.astype(np.float32)
	num_of_bins = bins.shape[0]
	digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
	b2 = np.ascontiguousarray(digitized).view(
		np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
	unique_array, unique_inverse_t, unique_counts = \
		np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
	p_ts = unique_counts / float(sum(unique_counts))
	PXs, PYs = np.asarray(pxs).T, np.asarray(pys1).T
	if calc_DKL:
		pxy_given_T = np.array(
			[calc_probs(i, unique_inverse_t, label, b, b1, len_unique_a) for i in range(0, len(unique_array))])
		p_XgT = np.vstack(pxy_given_T[:, 0])
		p_YgT = pxy_given_T[:, 1]
		p_YgT = np.vstack(p_YgT).T
		DKL_YgX_YgT = np.sum([inf_ut.KL(c_p_YgX, p_YgT.T) for c_p_YgX in p_YgX.T], axis=0)
		H_Xgt = np.nansum(p_XgT * np.log2(p_XgT), axis=1)
	local_IXT, local_ITY = calc_information_from_mat(PXs, PYs, p_ts, digitized, unique_inverse_x, unique_inverse_y,
	                                                 unique_array)
	return local_IXT, local_ITY


def calc_information_for_layer_with_other(data, bins, unique_inverse_x, unique_inverse_y, label,
                                          b, b1, len_unique_a, pxs, p_YgX, pys1):

	local_IXT, local_ITY = calc_information_sampling(data, bins, pys1, pxs, label, b, b1,
	                        len_unique_a, p_YgX, unique_inverse_x, unique_inverse_y)
	params = {}
	params['local_IXT'] = local_IXT
	params['local_ITY'] = local_ITY
	return params



def calc_information_for_epoch(iter_index, interval_information_display, ws_iter_index, bins, unique_inverse_x,
                               unique_inverse_y, label, b, b1,
                               len_unique_a, pxs, py_x, pys1):
	"""Calculate the information for all the layers for specific epoch"""
	params = np.array([calc_information_for_layer_with_other(data=ws_iter_index, bins=bins,
												unique_inverse_x=unique_inverse_x,
			                                    unique_inverse_y=unique_inverse_y, label=label, b=b,
			                                    b1=b1, len_unique_a=len_unique_a, pxs=pxs,
			                                    p_YgX=py_x, pys1=pys1)])

	if np.mod(iter_index, interval_information_display) == 0:
		print('Calculated The information of epoch number - {0}'.format(iter_index))
	return params


def extract_probs(label, x):
	"""calculate the probabilities of the given data and labels p(x), p(y) and (y|x)"""
	b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
	unique_array, unique_indices, unique_inverse_x, unique_counts = \
		np.unique(b, return_index=True, return_inverse=True, return_counts=True)
	unique_a = x[unique_indices]
	b1 = np.ascontiguousarray(unique_a).view(np.dtype((np.void, unique_a.dtype.itemsize * unique_a.shape[1])))
	pxs = unique_counts / float(np.sum(unique_counts))
	p_y_given_x = []
	for i in range(0, len(unique_array)):
		indexs = unique_inverse_x == i
		py_x_current = np.mean(label[indexs, :], axis=0)
		p_y_given_x.append(py_x_current)
	p_y_given_x = np.array(p_y_given_x).T
	return p_y_given_x, b1, b, unique_a, unique_inverse_x, pxs

def extract_probs_label(label,num_of_bins_y):
	data = np.squeeze(label.reshape(1, -1)[0])
	min = np.min(data)
	max = np.max(data)
	bins = np.linspace(min, max, num_of_bins_y)
	digitized = bins[np.digitize(data, bins) - 1].reshape(len(data), -1)
	b2 = np.ascontiguousarray(digitized).view(
		np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
	unique_array, unique_inverse_t, unique_counts = \
		np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
	p_ts = unique_counts / float(sum(unique_counts))
	return p_ts, unique_inverse_t, digitized

def get_information(ws_pub, ws_priv, ws_joint, x_pub, x_priv, x_joint, y, num_of_bins, num_of_bins_y,
					interval_information_display, calc_parallel=True, py_hats=0):
	"""Calculate the information for the network for all the epochs and all the layers"""
	print('Start calculating the information...')
	bins = np.linspace(-1, 1, num_of_bins)
	y = np.array(y).astype(np.float)
	pys1, unique_inverse_y,label = extract_probs_label(y,num_of_bins_y)
	p_y_given_x_pub, b1_pub, b_pub, unique_a_pub, unique_inverse_x_pub, pxs_pub = extract_probs(label, x_pub)
	p_y_given_x_priv, b1_priv, b_priv, unique_a_priv, unique_inverse_x_priv, pxs_priv = extract_probs(label, x_priv)
	p_y_given_x_joint, b1_joint, b_joint, unique_a_joint, unique_inverse_x_joint, pxs_joint = extract_probs(label, x_joint)
	# Shannon Entropy over label
	H2Label = -np.sum(pys1 * np.log2(pys1))
	# mutual Information between secret layer and label
	MI_pri_label = calc_information_for_inp_out(pxs_priv,pys1,label,unique_inverse_x_priv)
	# mutual Information between secret layer and label
	MI_pub_label = calc_information_for_inp_out(pxs_pub,pys1,label,unique_inverse_x_pub)

	if calc_parallel:
		print('calculating the information for public layer...')
		params_pub = np.array(Parallel(n_jobs=NUM_CORES)(delayed(calc_information_for_epoch)
		                            (i, interval_information_display, ws_pub[i], bins, unique_inverse_x_pub,
									unique_inverse_y, label, b_pub, b1_pub, len(unique_a_pub),
		        					pxs_pub, p_y_given_x_pub, pys1)
		                            for i in range(len(ws_pub))))
		print('calculating the information for secret layer...')
		params_priv = np.array(Parallel(n_jobs=NUM_CORES)(delayed(calc_information_for_epoch)
		                            (i, interval_information_display, ws_priv[i], bins, unique_inverse_x_priv,
									unique_inverse_y, label, b_priv, b1_priv, len(unique_a_priv),
		        					pxs_priv, p_y_given_x_priv, pys1)
		                            for i in range(len(ws_priv))))
		print('calculating the information for joint layer...')
		params_joint = np.array(Parallel(n_jobs=NUM_CORES)(delayed(calc_information_for_epoch)
		                            (i, interval_information_display, ws_joint[i], bins, unique_inverse_x_joint,
									unique_inverse_y, label, b_joint, b1_joint, len(unique_a_joint),
		        					pxs_joint, p_y_given_x_joint, pys1)
		                            for i in range(len(ws_joint))))

	else:
		params_pub = np.array([calc_information_for_epoch
								(i, interval_information_display, ws_pub[i], bins, unique_inverse_x_pub,
								unique_inverse_y, label, b_pub, b1_pub, len(unique_a_pub),
								pxs_pub, p_y_given_x_pub, pys1)
		                   		for i in range(len(ws_pub))])
		params_priv = np.array([calc_information_for_epoch
		                            (i, interval_information_display, ws_priv[i], bins, unique_inverse_x_priv,
									unique_inverse_y, label, b_priv, b1_priv, len(unique_a_priv),
		        					pxs_priv, p_y_given_x_priv, pys1)
		                            for i in range(len(ws_priv))])
		params_joint = np.array([calc_information_for_epoch
		                            (i, interval_information_display, ws_joint[i], bins, unique_inverse_x_joint,
									unique_inverse_y, label, b_joint, b1_joint, len(unique_a_joint),
		        					pxs_joint, p_y_given_x_joint, pys1)
		                            for i in range(len(ws_joint))])
	return params_pub, params_priv, params_joint, H2Label, MI_pri_label, MI_pub_label

def get_information_y_hat(x_pub, x_priv, x_joint, y_hat, num_of_bins_y):
	"""Calculate the information between public/secret inputs with y_hat"""
	print('Start calculating the information for y_hat...')
	y_hat = np.array(y_hat).astype(np.float)
	pys_hat, unique_inverse_y, y_hat = extract_probs_label(y_hat,num_of_bins_y)
	p_y_given_x_pub, b1_pub, b_pub, unique_a_pub, unique_inverse_x_pub, pxs_pub = extract_probs(y_hat, x_pub)
	p_y_given_x_priv, b1_priv, b_priv, unique_a_priv, unique_inverse_x_priv, pxs_priv = extract_probs(y_hat, x_priv)
	p_y_given_x_joint, b1_joint, b_joint, unique_a_joint, unique_inverse_x_joint, pxs_joint = extract_probs(y_hat, x_joint)
	# Shannon Entropy over label
	H2Y_hat = -np.sum(pys_hat * np.log2(pys_hat))
	# mutual Information between secret layer and label
	MI_pri_y_hat = calc_information_for_inp_out(pxs_priv,pys_hat,y_hat,unique_inverse_x_priv)
	# mutual Information between secret layer and label
	MI_pub_y_hat = calc_information_for_inp_out(pxs_pub,pys_hat,y_hat,unique_inverse_x_pub)
	return H2Y_hat, MI_pri_y_hat, MI_pub_y_hat

def get_information_y_hat(ws_priv, num_of_bins_y):
	"""Calculate the information containd in secret layer"""
	print('Start calculating the information contained in secret layer...')
	ws_priv_res = []
	for ws_p in ws_priv:
		ws_p_arr = np.array(ws_p).astype(np.float)
		min = np.min(ws_p_arr)
		max = np.max(ws_p_arr)
		pys_hat, _, _ = extract_probs_label(ws_p_arr, num_of_bins_y)
		H2Y_hat = -np.sum(pys_hat * np.log2(pys_hat))
		delta =  (max - min) / (num_of_bins_y)
		H2Y_hat += np.log2(delta)
		if H2Y_hat < 0:
			H2Y_hat = 0
		ws_priv_res.append(H2Y_hat)
	return ws_priv_res

def get_information_priv_y_hat(x_priv, y_hat, num_of_bins_y):
	"""Calculate the information between secret inputs with y_hat"""
	print('Start calculating the information for y_hat...')
	y_hat = np.array(y_hat).astype(np.float)
	pys_hat, unique_inverse_y, y_hat = extract_probs_label(y_hat,num_of_bins_y)
	p_y_given_x_priv, b1_priv, b_priv, unique_a_priv, unique_inverse_x_priv, pxs_priv = extract_probs(y_hat, x_priv)
	# mutual Information between secret layer and label
	MI_pri_y_hat = calc_information_for_inp_out(pxs_priv,pys_hat,y_hat,unique_inverse_x_priv)
	return MI_pri_y_hat.astype(np.float32)

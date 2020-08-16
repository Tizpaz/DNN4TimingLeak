from mat4py import *
import numpy as np
import sys
import os
import time

# def general(output):
# 	f= open(output,"a+")
#
# 	f.write("from gurobipy import *\n")
# 	f.write("import math\n")
# 	# f.write("import numpy as np\n")
# 	f.write("import time\n")
# 	f.write("import sys\n")
# 	f.write("\n")
#
# 	f.write("def multiply(w, x): \n")
# 	f.write("	return w * x \n")
# 	f.write("\n")
# 	f.close()

def parse_input_files(with_sec_file ,output, ith_bit, obj_str, k_bits_obs):
	data_with_sec = loadmat(with_sec_file)
	# print(data_with_sec)
	f= open(output,"w")

	f.write("from gurobipy import *\n")
	f.write("import math\n")
	# f.write("import numpy as np\n")
	f.write("import time\n")
	f.write("import sys\n")
	f.write("\n")

	f.write("def multiply(w, x): \n")
	f.write("	return w * x \n")
	f.write("\n")

	f.write("k_bits_obs = "+str(k_bits_obs)+"\n")
	f.write("M = 2000"+"\n")
	f.write("m = Model(\""+output+"_Model"+"\")\n")

	# First model with secret
	num_layer_pub = 0
	num_layer_sec = 0
	num_layer_joint = 0
	for (key, val) in data_with_sec.items():
		if key.startswith("W_priv"):
			num_layer_sec += 1

	# define first layer
	seen_fun = 0
	seen_fun_pub = []
	seen_fun_priv = []
	seen_fun_joint = []
	num_public_variables = 0
	num_priv_variables = 0
	for (key, val) in data_with_sec.items():
		if key.startswith("W_priv_enc_1"):
			s1 = np.matrix(val)
			lst_bias = data_with_sec["b_priv_enc_1"]
			dim_priv = np.shape(s1)
			num_priv_variables = dim_priv[0]
			for i in range(dim_priv[0]):
				f.write("x0"+ str(i) + " = m.addVar(vtype=GRB.BINARY, name=\"x0"+str(i)+"\")\n")
			f.write("\n")
			for i in range(dim_priv[0]):
				for j in range(dim_priv[1]):
					f.write("w"+ str(j+seen_fun) + "_" + str(i) + " = "+ str(round(s1[i,j],6)) +"\n")
			for j in range(len(lst_bias)):
				f.write("b"+ str(j+seen_fun) + " = "+ str(round(lst_bias[j],6)) +"\n")
			for j in range(dim_priv[1]):
				f.write("s0"+ str(j+seen_fun) + " = m.addVar(lb=-GRB.INFINITY, name=\"s0"+str(j+seen_fun)+"\")")
				f.write("\n")
			for j in range(dim_priv[1]):
				f.write("m.addConstr(s0"+ str(j+seen_fun) + " == ")
				for i in range(dim_priv[0]):
					if i < dim_priv[0] - 1:
						f.write("multiply(x0"+str(i)+",w"+ str(j+seen_fun) + "_" + str(i)+") + ")
					else:
						f.write("multiply(x0"+str(i)+",w"+ str(j+seen_fun) + "_" + str(i)+")")
				f.write(" + b"+ str(j+seen_fun) + ")")
				f.write("\n")
			for j in range(dim_priv[1]):
				f.write("d0"+ str(j+seen_fun) + " = m.addVar(vtype=GRB.BINARY, name=\"d0"+str(j+seen_fun)+"\")\n")
			for j in range(dim_priv[1]):
				f.write("f0"+ str(j+seen_fun) + " = m.addVar(vtype=GRB.BINARY, name=\"f0"+str(j+seen_fun)+"\")\n")
				f.write("g0"+ str(j+seen_fun) + " = m.addVar(lb = 0.0," + " name=\"g0"+str(j+seen_fun)+"\")\n")
				f.write("m.addConstr(g0"+ str(j+seen_fun) + " >= s0" + str(j+seen_fun)+")")
				f.write("\n")
				f.write("m.addConstr(g0"+ str(j+seen_fun) + " <= s0" + str(j+seen_fun) + " + (1 - d0"+ str(j+seen_fun)+") * M)")
				f.write("\n")
				f.write("m.addConstr(g0"+ str(j+seen_fun) + " <= d0"+ str(j+seen_fun)+"*M)")
				f.write("\n")
				f.write("m.addConstr(s0" + str(j+seen_fun) + " + (1 - d0"+ str(j+seen_fun)+") * M >= 0)")
				f.write("\n")
				f.write("m.addConstr(s0"+ str(j+seen_fun) + " - d0"+ str(j+seen_fun)+" * M <= 0)")
				f.write("\n")
				f.write("m.addConstr(f0"+ str(j+seen_fun) + " * (g0"+ str(j+seen_fun)+"- 0.5) >= 0)")
				f.write("\n")
				f.write("m.addConstr((1 - f0"+ str(j+seen_fun) + ") * (g0"+ str(j+seen_fun)+"- 0.5) <= 0)")
				f.write("\n")
			seen_fun_priv.append((seen_fun, seen_fun + dim_priv[1]))
			seen_fun += dim_priv[1]
			f.write("\n")


	max_layers = num_layer_sec
	for k in range(2,max_layers+1):
		for (key, val) in data_with_sec.items():
			if k <= num_layer_sec and key.startswith("W_priv_enc_"+str(k)):
				s1 = np.matrix(val)
				lst_bias = data_with_sec["b_priv_enc_"+str(k)]
				dim_priv = np.shape(s1)
				last_priv_front = seen_fun_priv[-1][0]
				last_priv_end = seen_fun_priv[-1][1]
				for i in range(dim_priv[0]):
					for j in range(dim_priv[1]):
						f.write("w"+ str(j+seen_fun) + "_" + str(i) + " = "+ str(round(s1[i,j],6)) +"\n")
				if isinstance(lst_bias, float):
					f.write("b"+ str(j+seen_fun) + " = "+ str(round(lst_bias,6)) +"\n")
				else:
					for j in range(len(lst_bias)):
						f.write("b"+ str(j+seen_fun) + " = "+ str(round(lst_bias[j],6)) +"\n")
				for j in range(dim_priv[1]):
					f.write("s0"+ str(j+seen_fun) + " = m.addVar(lb=-GRB.INFINITY,name=\"s0"+str(j+seen_fun)+"\")")
					f.write("\n")
				for j in range(dim_priv[1]):
					f.write("m.addConstr(s0"+ str(j+seen_fun) + " == ")
					for i in range(dim_priv[0]):
						if i < dim_priv[0] - 1:
							f.write("multiply(f0"+str(last_priv_front+i)+",w"+ str(j+seen_fun) + "_" + str(i)+") + ")
						else:
							f.write("multiply(f0"+str(last_priv_front+i)+",w"+ str(j+seen_fun) + "_" + str(i)+")")
					f.write(" + b"+ str(j+seen_fun) + ")")
					f.write("\n")
				for j in range(dim_priv[1]):
					f.write("d0"+ str(j+seen_fun) + " = m.addVar(vtype=GRB.BINARY, name=\"d0"+str(j+seen_fun)+"\")\n")
				for j in range(dim_priv[1]):
					f.write("f0"+ str(j+seen_fun) + " = m.addVar(vtype=GRB.BINARY, name=\"f0"+str(j+seen_fun)+"\")\n")
					f.write("g0"+ str(j+seen_fun) + " = m.addVar(lb = 0.0," + " name=\"g0"+str(j+seen_fun)+"\")\n")
					f.write("m.addConstr(g0"+ str(j+seen_fun) + " >= s0" + str(j+seen_fun)+")")
					f.write("\n")
					f.write("m.addConstr(g0"+ str(j+seen_fun) + " <= s0" + str(j+seen_fun) + " + (1 - d0"+ str(j+seen_fun)+") * M)")
					f.write("\n")
					f.write("m.addConstr(g0"+ str(j+seen_fun) + " <= d0"+ str(j+seen_fun)+" * M)")
					f.write("\n")
					f.write("m.addConstr(s0" + str(j+seen_fun) + " + (1 - d0"+ str(j+seen_fun)+") * M >= 0)")
					f.write("\n")
					f.write("m.addConstr(s0"+ str(j+seen_fun) + " - d0"+ str(j+seen_fun)+" * M <= 0)")
					f.write("\n")
					f.write("m.addConstr(f0"+ str(j+seen_fun) + " * (g0"+ str(j+seen_fun)+"- 0.5) >= 0)")
					f.write("\n")
					f.write("m.addConstr((1 - f0"+ str(j+seen_fun) + ") * (g0"+ str(j+seen_fun)+"- 0.5) <= 0)")
					f.write("\n")

				if k == num_layer_sec and key.startswith("W_priv_enc_"+str(k)):
					for j in range(dim_priv[1]):
						if k_bits_obs[j] != -1:
							f.write("m.addConstr(f0"+ str(j+seen_fun) + " == k_bits_obs["+str(j)+"])\n")
					f.write("m.setObjective(")
					pow = 0
					for j in range(dim_priv[1]):
						if k_bits_obs[j] == -1:
							if j < dim_priv[1] - 1:
								f.write(str(2**pow) + " * f0" + str(j+seen_fun) + " + ")
							else:
								f.write(str(2**pow) + " * f0" + str(j+seen_fun))
							pow += 1
					f.write(", " + obj_str+")\n")
				seen_fun_priv.append((seen_fun, seen_fun + dim_priv[1]))
				seen_fun += dim_priv[1]
				f.write("\n")
	# f.write("m.Params.PoolSearchMode=0\n")
	# for phonemaster and passmatcher2
	# f.write("m.Params.timeLimit=10.0\n")
	# micro-benchmark is 100 (the gabfeed = 10000, SnapBuddy 60, PassCheck = 10000, PassCheck2 = 10000, phonemaster = 100)
	# f.write("m.Params.PoolSolutions=1\n")
	f.write("startTime = int(round(time.time() * 1000))\n\n")
	f.write("m.optimize()\n\n")
	f.write("endTime = int(round(time.time() * 1000))\n")
	f.write("rTime = endTime - startTime\n")
	f.write("print(\"Time taken to calculate (in milli-seconds):\")\n")
	f.write("print(rTime)\n")
	f.write("f2 = open(\""+output+"_Model_res_obj"+".txt\",\'a+\')\n")
	f.write("f2.write(\'Time taken to analyze: \' + str(rTime))\n")
	f.write("f2.write(\'\\n\')\n")
	f.write("if m.SolCount > 0:\n")
	f.write("   f2.write(\'Objective Value: \')\n")
	f.write("   f2.write(str(m.objVal))\n")
	f.write("else:\n")
	f.write("   f2.write(\'Objective Value: \')\n")
	f.write("f2.write(\'\\n\')\n")
	f.write("f2.close()\n")
	f.close()


def parse_input_files_2(with_sec_file, output, k_bits_obs):

	f = open(output,"w")

	f.write("from gurobipy import *\n")
	f.write("import math\n")
	# f.write("import numpy as np\n")
	f.write("import time\n")
	f.write("import sys\n")
	f.write("\n")

	f.write("def multiply(w, x): \n")
	f.write("	return w * x \n")
	f.write("\n")

	data_with_sec = loadmat(with_sec_file)
	# print(data_with_sec)

	f.write("k_bits_obs = "+str(k_bits_obs)+"\n")
	f.write("M = 10000"+"\n")
	f.write("m = Model(\""+output+"_Model"+"\")\n")

	# First model with secret
	num_layer_pub = 0
	num_layer_sec = 0
	num_layer_joint = 0
	for (key, val) in data_with_sec.items():
		if key.startswith("W_priv"):
			num_layer_sec += 1

	# define first layer
	seen_fun = 0
	seen_fun_pub = []
	seen_fun_priv = []
	seen_fun_joint = []
	num_public_variables = 0
	num_priv_variables = 0
	for (key, val) in data_with_sec.items():
		if key.startswith("W_priv_enc_1"):
			s1 = np.matrix(val)
			lst_bias = data_with_sec["b_priv_enc_1"]
			dim_priv = np.shape(s1)
			num_priv_variables = dim_priv[0]
			for i in range(dim_priv[0]):
				f.write("x0"+ str(i) + " = m.addVar(vtype=GRB.BINARY, name=\"x0"+str(i)+"\")\n")
			f.write("\n")
			for i in range(dim_priv[0]):
				for j in range(dim_priv[1]):
					f.write("w"+ str(j+seen_fun) + "_" + str(i) + " = "+ str(round(s1[i,j],6)) +"\n")
			for j in range(len(lst_bias)):
				f.write("b"+ str(j+seen_fun) + " = "+ str(round(lst_bias[j],6)) +"\n")
			for j in range(dim_priv[1]):
				f.write("s0"+ str(j+seen_fun) + " = m.addVar(lb=-GRB.INFINITY, name=\"s0"+str(j+seen_fun)+"\")")
				f.write("\n")
			for j in range(dim_priv[1]):
				f.write("m.addConstr(s0"+ str(j+seen_fun) + " == ")
				for i in range(dim_priv[0]):
					if i < dim_priv[0] - 1:
						f.write("multiply(x0"+str(i)+",w"+ str(j+seen_fun) + "_" + str(i)+") + ")
					else:
						f.write("multiply(x0"+str(i)+",w"+ str(j+seen_fun) + "_" + str(i)+")")
				f.write(" + b"+ str(j+seen_fun) + ")")
				f.write("\n")
			for j in range(dim_priv[1]):
				f.write("d0"+ str(j+seen_fun) + " = m.addVar(vtype=GRB.BINARY, name=\"d0"+str(j+seen_fun)+"\")\n")
			for j in range(dim_priv[1]):
				f.write("f0"+ str(j+seen_fun) + " = m.addVar(vtype=GRB.BINARY, name=\"f0"+str(j+seen_fun)+"\")\n")
				f.write("g0"+ str(j+seen_fun) + " = m.addVar(lb = 0.0," + " name=\"g0"+str(j+seen_fun)+"\")\n")
				f.write("m.addConstr(g0"+ str(j+seen_fun) + " >= s0" + str(j+seen_fun)+")")
				f.write("\n")
				f.write("m.addConstr(g0"+ str(j+seen_fun) + " <= s0" + str(j+seen_fun) + " + (1 - d0"+ str(j+seen_fun)+") * M)")
				f.write("\n")
				f.write("m.addConstr(g0"+ str(j+seen_fun) + " <= d0"+ str(j+seen_fun)+"*M)")
				f.write("\n")
				f.write("m.addConstr(s0" + str(j+seen_fun) + " + (1 - d0"+ str(j+seen_fun)+") * M >= 0)")
				f.write("\n")
				f.write("m.addConstr(s0"+ str(j+seen_fun) + " - d0"+ str(j+seen_fun)+" * M <= 0)")
				f.write("\n")
				f.write("m.addConstr(f0"+ str(j+seen_fun) + " * (g0"+ str(j+seen_fun)+"- 0.5) >= 0)")
				f.write("\n")
				f.write("m.addConstr((1 - f0"+ str(j+seen_fun) + ") * (g0"+ str(j+seen_fun)+"- 0.5) <= 0)")
				f.write("\n")
			seen_fun_priv.append((seen_fun, seen_fun + dim_priv[1]))
			seen_fun += dim_priv[1]
			f.write("\n")


	max_layers = num_layer_sec
	for k in range(2,max_layers+1):
		for (key, val) in data_with_sec.items():
			if k <= num_layer_sec and key.startswith("W_priv_enc_"+str(k)):
				s1 = np.matrix(val)
				lst_bias = data_with_sec["b_priv_enc_"+str(k)]
				dim_priv = np.shape(s1)
				last_priv_front = seen_fun_priv[-1][0]
				last_priv_end = seen_fun_priv[-1][1]
				for i in range(dim_priv[0]):
					for j in range(dim_priv[1]):
						f.write("w"+ str(j+seen_fun) + "_" + str(i) + " = "+ str(round(s1[i,j],6)) +"\n")
				if isinstance(lst_bias, float):
					f.write("b"+ str(j+seen_fun) + " = "+ str(round(lst_bias,6)) +"\n")
				else:
					for j in range(len(lst_bias)):
						f.write("b"+ str(j+seen_fun) + " = "+ str(round(lst_bias[j],6)) +"\n")
				for j in range(dim_priv[1]):
					f.write("s0"+ str(j+seen_fun) + " = m.addVar(lb=-GRB.INFINITY,name=\"s0"+str(j+seen_fun)+"\")")
					f.write("\n")
				for j in range(dim_priv[1]):
					f.write("m.addConstr(s0"+ str(j+seen_fun) + " == ")
					for i in range(dim_priv[0]):
						if i < dim_priv[0] - 1:
							f.write("multiply(f0"+str(last_priv_front+i)+",w"+ str(j+seen_fun) + "_" + str(i)+") + ")
						else:
							f.write("multiply(f0"+str(last_priv_front+i)+",w"+ str(j+seen_fun) + "_" + str(i)+")")
					f.write(" + b"+ str(j+seen_fun) + ")")
					f.write("\n")
				for j in range(dim_priv[1]):
					f.write("d0"+ str(j+seen_fun) + " = m.addVar(vtype=GRB.BINARY, name=\"d0"+str(j+seen_fun)+"\")\n")
				for j in range(dim_priv[1]):
					f.write("f0"+ str(j+seen_fun) + " = m.addVar(vtype=GRB.BINARY, name=\"f0"+str(j+seen_fun)+"\")\n")
					f.write("g0"+ str(j+seen_fun) + " = m.addVar(lb = 0.0," + " name=\"g0"+str(j+seen_fun)+"\")\n")
					f.write("m.addConstr(g0"+ str(j+seen_fun) + " >= s0" + str(j+seen_fun)+")")
					f.write("\n")
					f.write("m.addConstr(g0"+ str(j+seen_fun) + " <= s0" + str(j+seen_fun) + " + (1 - d0"+ str(j+seen_fun)+") * M)")
					f.write("\n")
					f.write("m.addConstr(g0"+ str(j+seen_fun) + " <= d0"+ str(j+seen_fun)+" * M)")
					f.write("\n")
					f.write("m.addConstr(s0" + str(j+seen_fun) + " + (1 - d0"+ str(j+seen_fun)+") * M >= 0)")
					f.write("\n")
					f.write("m.addConstr(s0"+ str(j+seen_fun) + " - d0"+ str(j+seen_fun)+" * M <= 0)")
					f.write("\n")
					f.write("m.addConstr(f0"+ str(j+seen_fun) + " * (g0"+ str(j+seen_fun)+"- 0.5) >= 0)")
					f.write("\n")
					f.write("m.addConstr((1 - f0"+ str(j+seen_fun) + ") * (g0"+ str(j+seen_fun)+"- 0.5) <= 0)")
					f.write("\n")

				if k == num_layer_sec and key.startswith("W_priv_enc_"+str(k)):
					for j in range(dim_priv[1]):
						f.write("m.addConstr(f0"+ str(j+seen_fun) + " == k_bits_obs["+str(j)+"])\n")
					f.write("m.setObjective(")
					for j in range(dim_priv[1]):
						if j < dim_priv[1] - 1:
							f.write("f0"+ str(j+seen_fun) + " + ")
						else:
							f.write("f0"+ str(j+seen_fun))
					f.write(", GRB.MAXIMIZE)\n")
				seen_fun_priv.append((seen_fun, seen_fun + dim_priv[1]))
				seen_fun += dim_priv[1]
				f.write("\n")
	f.write("m.Params.PoolSearchMode=2\n")
	# for phonemaster and passmatcher2
	# f.write("m.Params.timeLimit=10.0\n")
	# micro-benchmark is 100 (the gabfeed = 10000, SnapBuddy 60, PassCheck = 10000, PassCheck2 = 10000, phonemaster = 100)
	f.write("m.Params.PoolSolutions=100\n")
	f.write("startTime = int(round(time.time() * 1000))\n\n")
	f.write("m.optimize()\n\n")
	f.write("endTime = int(round(time.time() * 1000))\n")
	f.write("rTime = endTime - startTime\n")
	f.write("print(\"Time taken to calculate (in milli-seconds):\")\n")
	f.write("print(rTime)\n")
	f.write("f1 = open(\""+output+"_Model_res"+".txt\",\'a\')\n")
	f.write("f1.write(\'Time taken to analyze: \' + str(rTime))\n")
	f.write("f1.write(\'\\n\')\n")
	f.write("f1.write(\'number of solutions: \' + str(m.SolCount))\n")
	f.write("f1.write(\'\\n\')\n")
	f.write("for k in range(m.SolCount):\n")
	f.write("   f1.write(\'Solution Number: \' + str(k))\n")
	f.write("   f1.write(\'\\n\')\n")
	f.write("   m.Params.solutionNumber = k\n")
	f.write("   for d in m.getVars():\n")
	f.write("      if d.varName.startswith(\'x\'):\n")
	f.write("         name = d.varName\n")
	f.write("         val = d.xn\n")
	f.write("         f1.write(str(name)+\" \")\n")
	f.write("         f1.write(str(val))\n")
	f.write("         f1.write(\'\\n\')\n")
	f.write("f1.write(\'---------------------------\\n\')\n")
	f.write("f1.close()\n")
	f.close()


# def parse_input_files_3(with_sec_file, output, k_bits_obs):
#
# 	f1 = open(output+"_Model_res"+".txt","a")
# 	f1.write("Time taken to analyze: 0\n")
# 	f1.write("number of solutions: 1\n")
# 	# f1.write(str(k_bits_obs)+"\n")
# 	f1.write("---------------------------\n")
# 	f1.close()


def fun_calculate(file_with_sec, output_file, output_file_final, file_to_read, i, k_bits_obs, lst_str_binary, results_lists, visited_sol):

	if i > k_bits_obs - 1:
		return

	parse_input_files(file_with_sec, output_file, i, "GRB.MINIMIZE", lst_str_binary)
	os.system("gurobi.sh " + output_file + " > tmp.out")

	with open(file_to_read) as f:
		content = f.readlines()

	for x in content:
		x = x.strip()
		if "Time" in x:
			continue
		if "Objective Value" in x:
			if len(x.split()) <= 2:
				f3 = open(output_file+"_Model_res_obj"+".txt","w")
				f3.close()
				return


	parse_input_files(file_with_sec, output_file, i, "GRB.MAXIMIZE", lst_str_binary)
	os.system("gurobi.sh " + output_file + " > tmp.out")
        # print("here1")
	lst_str_binary_min = []
	for val in lst_str_binary:
		lst_str_binary_min.append(val)

	lst_str_binary_max = []
	for val in lst_str_binary:
		lst_str_binary_max.append(val)

	with open(file_to_read) as f:
		content = f.readlines()

	turn = 0
	time_comp = 0
	for x in content:
		x = x.strip()
		if "Time" in x:
			val_time_str = x.split(":")[1]
			time_comp += float(val_time_str)
			continue
		if "Objective Value" in x and turn == 0:
			if len(x.split()) > 2:
				val1 = x.split()[2]
			else:
				f3 = open(output_file+"_Model_res_obj"+".txt","w")
				f3.close()
                                # print("here2")
				return
			turn = 1
		elif "Objective Value" in x and turn == 1:
			if len(x.split()) > 2:
				val2 = x.split()[2]
			else:
				f3 = open(output_file+"_Model_res_obj"+".txt","w")
				f3.close()
				return
			if int(abs(float(val1))) == int(abs(float(val2))):
				min_val = int(abs(float(val1)))
				while i <= k_bits_obs - 1:
					lst_str_binary_min[i] = min_val % 2
					lst_str_binary_max[i] = lst_str_binary_min[i]
					i = i + 1
					min_val = min_val / 2

				if str(lst_str_binary_min) in visited_sol and str(lst_str_binary_max) in visited_sol:
					return
				print("Time (min-max) in this step:" + val_time_str)
				f3 = open(output_file+"_Model_res_obj"+".txt","w")
				f3.close()
				visited_sol.add(str(lst_str_binary_min))
				visited_sol.add(str(lst_str_binary_max))
				parse_input_files_2(file_with_sec, output_file_final, lst_str_binary_min)
				os.system("gurobi.sh " + output_file_final + " > tmp.out")

			else:
				lst_str_binary_min[i] = int(abs(float(val1))) % 2
				lst_str_binary_max[i] = 1 - lst_str_binary_min[i]
				if str(lst_str_binary_min) in visited_sol and str(lst_str_binary_max) in visited_sol:
					return
				f3 = open(output_file+"_Model_res_obj"+".txt","w")
				f3.close()
				print("Time (min-max) in this step:" + val_time_str)
				visited_sol.add(str(lst_str_binary_min))
				visited_sol.add(str(lst_str_binary_max))
				# print(val1)
				# print(val2)
				# print("Two Solutions: " + str(lst_str_binary_min) + " " + str(lst_str_binary_max))
				if  abs(int(abs(float(val1))) - int(abs(float(val2)))) == 1:
					min_val = int(abs(float(val1)))
					i = i + 1
					while i <= k_bits_obs - 1:
						min_val = min_val / 2
						lst_str_binary_min[i] = min_val % 2
						lst_str_binary_max[i] = lst_str_binary_min[i]
						i = i + 1
					parse_input_files_2(file_with_sec, output_file_final, lst_str_binary_min)
					os.system("gurobi.sh " + output_file_final + " > tmp.out")
					parse_input_files_2(file_with_sec, output_file_final, lst_str_binary_max)
					os.system("gurobi.sh " + output_file_final + " > tmp.out")
				elif i == k_bits_obs - 1:
					# print("total time (so far): " + str(time_comp))
					parse_input_files_2(file_with_sec, output_file_final, lst_str_binary_min)
					os.system("gurobi.sh " + output_file_final + " > tmp.out")
					parse_input_files_2(file_with_sec, output_file_final, lst_str_binary_max)
					os.system("gurobi.sh " + output_file_final + " > tmp.out")
					# results_lists.append(lst_str_binary_min)
					# results_lists.append(lst_str_binary_max)
				else:
					fun_calculate(file_with_sec, output_file, output_file_final, file_to_read, i+1, k_bits_obs, lst_str_binary_min, results_lists, visited_sol)
					fun_calculate(file_with_sec, output_file, output_file_final, file_to_read, i+1, k_bits_obs, lst_str_binary_max, results_lists, visited_sol)

			turn = 0
			f3 = open(output_file+"_Model_res_obj"+".txt","w")
			f3.close()

if __name__ == '__main__':
	file_with_sec = sys.argv[1]
	output_file = sys.argv[2]
	# file_to_read = sys.argv[3]
	k_bits_obs = int(sys.argv[3]) 	# num. bits in output layer

	# general(output_file)
	file_to_read = output_file + "_Model_res_obj"+".txt"
	lst_str_binary = []

	splt = output_file.split(".")[:-1]
	output_file_final = ''.join(splt)
	output_file_final = output_file_final + "_MIP.py"

	# general_parse_input(output_file_final)

	for j in range(k_bits_obs):
		lst_str_binary.append(-1)

	results_lists = []
	visited_sol = set()

	fun_calculate(file_with_sec, output_file, output_file_final, file_to_read, 0, k_bits_obs, lst_str_binary, results_lists, visited_sol)

	# visited_val = set()
	# final_results_lists = []
	#
	# for lst in results_lists:
	# 	power = 0
	# 	res1 = 0
	# 	for val in lst:
	# 		res1 += (2 ** power) * val
	# 		power += 1
	# 	if res1 not in visited_val:
	# 		final_results_lists.append(lst)
	# 	visited_val.add(res1)
	#
	# f1 = open(file_to_read,"w")
	# for val in final_results_lists:
	# 	f1.write(str(val))
	# 	f1.write("\n")

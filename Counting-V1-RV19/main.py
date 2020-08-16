from mat4py import *
import numpy as np
import sys


def general(output):
	f= open(output,"a+")

	f.write("from gurobipy import *\n")
	f.write("import math\n")
	# f.write("import numpy as np\n")
	f.write("import time\n")
	f.write("import sys\n")
	f.write("\n")

	f.write("def multiply(w, x): \n")
	f.write("	return w * x \n")
	f.write("\n")
	f.close()	

def parse_input_files(with_sec_file ,output, k_bits_obs):
	data_with_sec = loadmat(with_sec_file)
	# print(data_with_sec)
	f= open(output,"a+")

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
	f.write("m.Params.timeLimit=10.0\n")
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

if __name__ == '__main__':
	file_with_sec = sys.argv[1]
	output_file = sys.argv[2]
	k_bits_obs = int(sys.argv[3]) 	# num. bits in output layer
	general(output_file)
	for i in range(2**k_bits_obs):
		str_binary = "{0:0" + str(k_bits_obs) + "b}"
		str_binary = str_binary.format(i)
		lst_str_binary = []
		for x in str_binary:
			lst_str_binary.append(int(x))
		parse_input_files(file_with_sec, output_file, lst_str_binary)

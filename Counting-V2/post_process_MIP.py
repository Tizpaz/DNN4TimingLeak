import numpy as np
import sys

if __name__ == '__main__':
	input_file = sys.argv[1]
	f = open(input_file,"r")
	shannon_entropy = 0.0
	guessing_entropy = 0.0
	total = 0
	total_time = 0
	num_class_obs = 0
	for line in f:
		line = line.strip()
		if "number of solutions" in line:
			val = int(line.split()[3])
			# > 0 for RegEx
			# > 1 for Branch_Loop and all applications
			if val > 0:
				shannon_entropy += val * np.log2(val)
				guessing_entropy += val * val
				num_class_obs += 1
				total += val
		if "Time taken to analyze" in line:
			val = int(line.split()[4])
			total_time += val
	shannon_entropy = (shannon_entropy/total)
	guessing_entropy = (guessing_entropy/(2*total))
	print("total time : " + str(total_time))
	print("Num. Class of Observations : " + str(num_class_obs))
	print("Shannon Entropy : " + str(shannon_entropy))
	print("Guessing Entropy : " + str(guessing_entropy))

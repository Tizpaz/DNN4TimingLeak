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
        min_entropy = sys.maxint
	for line in f:
		line = line.strip()
		if "number of solutions" in line:
			val = int(line.split()[3])
			# > 0 for RegEx
			# > 1 for Branch_Loop and all applications
			if val > 1:
				shannon_entropy += val * np.log2(val)
				guessing_entropy += val * val
				num_class_obs += 1
                                min_entropy = min(min_entropy,val)
				total += val
		if "Time taken to analyze" in line:
			val = int(line.split()[4])
			total_time += val
	shannon_entropy = (shannon_entropy/total)
	guessing_entropy = (guessing_entropy/(2*total))
	min_entropy = np.log2(total) - np.log2(num_class_obs)
	print("total time : " + str(total_time))
	print("Min entropy : " + str(min_entropy))
	print("Shannon Entropy : " + str(shannon_entropy))
	print("Guessing Entropy : " + str(guessing_entropy))
        print("Min-Guess Entropy : " + str((float(min_entropy)+1.0)/2.0))

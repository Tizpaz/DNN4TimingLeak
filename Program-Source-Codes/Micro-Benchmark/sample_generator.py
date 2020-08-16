# You need to specify the program to run,
# also the range of secret and public inputs.
# This is a simple for B_L_1
# After input generation, please add column names to the csv files: p0 for the first colmn and s0...sk for the rest execpt time for the last column
import os
import random
program_to_run = 'B_L_1'

os.system('javac ' + program_to_run + '.java')
secret_range = range(25)
public_range = range(100)

for i in range(756):
    cur_secret = random.choice(secret_range)
    for j in public_range:
        os.system('java ' + 'B_L_1 ' + str(cur_secret) + ' ' + str(j))

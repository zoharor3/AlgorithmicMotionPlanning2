import re

with open ("results_prob_0.2.txt", "rt") as myfile:  # Open lorem.txt for reading text
    contents = myfile.read()              # Read the entire file to a string
print(contents)

total_cost = re.finditer("Total cost of path:", contents)

cost = []
for res in total_cost:
    index = res.regs[0][1]
    cost.append(float(contents[index+1:index+6]))

print('average cost is:', sum(cost)/len(cost))

total_time = re.finditer("Total time:", contents)

time = []
for res in total_time:
    index = res.regs[0][1]
    time.append(float(contents[index+1:index+6]))

print('average time is:', sum(time)/len(time))
import subprocess

n_iter = 10

for i in range(n_iter):
    subprocess.call("python run.py -map map_mp.json -task mp -ext_mode E1 -goal_prob 0.2", shell=True)
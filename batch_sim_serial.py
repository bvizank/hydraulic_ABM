from run_sim import run_sim
import os

for i in range(30):
    run_sim(id=i, days=90, bbn_models=[])
    curr_dir = os.getcwd()
    files_in_dir = os.listdir(curr_dir)
    
    for file in files_in_dir:
        if file.endswith('.rpt') or file.endswith('.bin') or file.endswith('.out'):
            os.remove(os.path.join(curr_dir, file))
import os
import wandb

import os.path as osp
from datetime import datetime

# Initialize API
api = wandb.Api()

# Define the workspace and project
workspace = "eb-lab"
project = "learned_aggregation_meta_train"  # Replace with your project's name

compare_date_str = "2023-08-12T00:00:00.000Z"
compare_date = datetime.strptime(compare_date_str, "%Y-%m-%dT%H:%M:%S.%fZ")

def parse_date(date_str):
    """Parse a date string which might or might not have fractional seconds."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
    
    
tested = {
    'fedlagg':{(h,ll): False for h in [4,8,16,32] for ll in [1.0,0.5,0.3,0.1]},
    'fedlopt':{(h,ll): False for h in [4,8,16,32] for ll in [1.0,0.5,0.3,0.1]},
    'fedlagg-adafac':{(h,ll): False for h in [4,8,16,32] for ll in [1.0,0.5,0.3,0.1]}
}

config_map = {
    'fedlagg':'config/meta_test/small-image-mlp-fmst_fedlagg.py',
    'fedlopt':'config/meta_test/small-image-mlp-fmst_fedlopt.py',
    'fedlagg-adafac':'config/meta_test/small-image-mlp-fmst_fedlagg-adafac.py'
}

def get_checkpoint(files,download_dir='/tmp'):
    ckpts = [x for x in files if 'global_step' in x.name]
    if len(ckpts) > 1:
        print(ckpts)
        
    assert len(ckpts) <= 1, "multiple checkpoints exist can't determine which one to use"
    
    if len(ckpts) == 0:
        return None
    
    ckpts[0].download('/tmp',replace=True)
    return osp.join('/tmp',ckpts[0].name)
    
    
cmd_prefix = "CUDA_VISIBLE_DEVICES=1 "
runs = api.runs(f"{workspace}/{project}")
for i,run in enumerate(runs):
        
    # print(run.created_at)
    run_date = parse_date(run.created_at)
    if run_date > compare_date and run.config['optimizer'] in list(config_map.keys()):
        
        # print(run.created_at)
        h = run.config['num_local_steps']
        llr = run.config['local_learning_rate']
        opt = run.config['optimizer']
        
        try:
            tested[opt][(h,llr)]
        except KeyError:
            continue
        
        if tested[opt][(h,llr)]:
            continue
            
        ckpt_name = get_checkpoint(run.files(),download_dir='/tmp')
        
        if ckpt_name is None: 
             continue
        
        command = cmd_prefix + 'python src/main.py --config {} --name_suffix {} --local_learning_rate {} --num_local_steps {} --test_checkpoint {}'.format(
            config_map[opt],"_5K_iters_H={}_llr={}".format(h,llr),llr,h,ckpt_name)
        
        print(command)
        os.system(command)
        
        tested[opt][(h,llr)] = True
        
    else:
        continue
        
        
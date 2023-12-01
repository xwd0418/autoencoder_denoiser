from experiment import Experiment
import sys, torch
import random, numpy as np
from transfer_w_cross_validation import DenoiseExp

# Main Driver for your code. Either run `python main.py` which will run the experiment with default config
# or specify the configuration by running `python main.py custom`
if __name__ == "__main__":
    exp_name = 'default'
    
    seed = 10086
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if len(sys.argv) > 2:
        exp_folder = sys.argv[1]
        exp_name = sys.argv[2]
    else:
        raise Exception(" missing sys args")

    print("Running Experiment: ", exp_name)
    exp = Experiment(exp_folder, exp_name)
    exp.run()
    config = exp.test() 
    
    TransferLearningExp = DenoiseExp(exp_name, config)
    k_fold = 2 if config.get("DEBUG") else 10
    TransferLearningExp.run(k_fold=k_fold)

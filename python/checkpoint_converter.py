import torch
import pickle5 as pickle

## Checkpoint Path
path = None 

state = None
with open(path, "rb") as f:
    state = pickle.load(f)
    print(state["metadata"])

    ## load txt file and convert to text. For syncing with the changed file name
    # path_2 = "env.txt"
    # with open(path_2, "r") as ff:
    #     text = ff.read()
    #     state["metadata"] = text

    ## for cascading learning
    state["muscle_optimizer"] = None  
    
with open(path + '_revised', 'wb') as f:
    pickle.dump(state, f)

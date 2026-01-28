import torch
import numpy as np
import yaml
import utils
from utils import select_llm, read_file, generate, safe_load_pickle, select_layers_to_steer, remove_junk, get_coefs
import pickle
import gc
import sys
from tqdm import tqdm
from args import get_args
import os

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

num_gpus = torch.cuda.device_count()
print("Number of GPUs available:", num_gpus)



rep_token, model_type, dataset_label, method, version, label = get_args()
assert label in ['soft', 'hard']
use_soft_labels = label=='soft'

print(f"rep_token = {rep_token}")
print(f"model_name = {model_type}")
print(f"dataset_label = {dataset_label}")
print(f"control_method = {method}")
print(f"use soft labels = {use_soft_labels}")

if dataset_label=='jailbreaking':
    max_tokens = 200
else:
    max_tokens = 50



COEFS = get_coefs(model_type, use_soft_labels)


dataset_to_lower = {'fears':True, 
                    'personalities':True, 
                    'moods':True, 
                    'places':False, 
                    'personas':False,
                    'jailbreaking': False,
                     'custom':False}
def main():

    torch.backends.cudnn.benchmark = True        
    torch.backends.cuda.matmul.allow_tf32 = True        

    
    concepts_fname = f'data/concepts/{dataset_label}.txt'  
    concepts = read_file(concepts_fname, lower=dataset_to_lower[dataset_label])
    
    
    with open("test_prompts.yaml", "r") as infile:
        test_prompts_dict = yaml.safe_load(infile)  
    prompt = test_prompts_dict[dataset_label][int(version)]
    
    outpath = utils.get_steered_output_filename(method, dataset_label, rep_token, model_type, version, use_soft_labels)

    # p = safe_load_pickle(outpath)
    # if p is None: #output doesn't exist
    #     llm = select_llm(model_type)
    #     layers_to_control = list(range(1,llm.language_model.config.num_hidden_layers,1))
    #     starting_idx = 0
    #     all_outputs = {}
    # else: #output exists
    #     is_val, prob = utils.validate_output_dict(p, concepts)
    #     if is_val: 
    #         print(f"output exists: {dataset_label}{rep_token} {method} {model_type} version {version}")
    #         return
    #     else: #output is incomplete
    #         print(prob)
    #         llm = select_llm(model_type)
    #         layers_to_control = list(range(1,llm.language_model.config.num_hidden_layers,1))
    #         starting_idx = 0
    #         all_outputs = {}

    llm = select_llm(model_type)
    starting_idx = 0  ###########
    all_outputs = {}   ##########
    layers_to_control = list(range(1,llm.language_model.config.num_hidden_layers,1))############################
    
    for concept_idx, concept in enumerate(tqdm(concepts[starting_idx:])):
        print(f"=============================================CONCEPT={concept} ============================================")
        print("Layers to control: ", layers_to_control)
        all_outputs[concept] = {}

        outputs = generate(concept, llm, prompt, use_soft_labels = use_soft_labels,
                           coefs=COEFS, rep_token = rep_token,
                        control_method=method, max_tokens=max_tokens, 
                           gen_orig=True, hidden_state = 'block', 
                           layers_to_control = layers_to_control,
                          start_from_token = 0, head_agg = 'mean')

        all_outputs[concept]= outputs

        assert len(all_outputs[concept]) == len(COEFS)
    with open(outpath, "wb") as file:
        pickle.dump(all_outputs, file)
    del llm
    torch.cuda.empty_cache()
    gc.collect()

                
if __name__ == "__main__":
    main()   

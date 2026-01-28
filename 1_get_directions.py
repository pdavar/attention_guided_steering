import torch
import numpy as np
from datasets import *
from utils import select_llm, read_file, compute_save_directions, get_tokenidx_per_layer_per_concept
import generation_utils
import sys
from args import get_args

rep_token, model_name, concept_type, method, version, label = get_args()

print(f"rep_token = {rep_token}")
print(f"model_name = {model_name}")
print(f"concept_type = {concept_type}")
print(f"control_method = {method}")
print(f"labels = {label}")
assert label in ['hard', 'soft']



paired_samples = False
use_soft_labels = label=='soft'
datasize = 'single'
ATTN_DIR = "data/attention_to_prompt"
                    
def main(model_name, concept_type):

    torch.backends.cudnn.benchmark = True        
    torch.backends.cuda.matmul.allow_tf32 = True    
    fname = f"data/concepts/{concept_type}.txt"
    dataset_to_lower = {'fears':True, 
                    'personalities':True, 
                    'moods':True, 
                    'places':False, 
                    'personas':False,
                    'jailbreaking': False,
                     'custom':False}
                     
    llm = select_llm(model_name)
                     
                     
    concept_list = read_file(fname, lower=dataset_to_lower[concept_type])
    dataset_fn = get_dataset_fn(concept_type, paired_samples = paired_samples)
    
    for concept in concept_list:
        print(f"==== {concept} =====")
        if rep_token == 'max_attn_per_layer':
            layer_to_token = get_tokenidx_per_layer_per_concept(concept, model_name, head_agg = 'mean', root_dir = ATTN_DIR)
            print("layer to token: ", layer_to_token)
        else: layer_to_token = None
        
        data = dataset_fn(llm, concept,datasize) 
        
        compute_save_directions(llm, data, use_soft_labels, concept, rep_token = rep_token,
                                hidden_state = 'block', layer_to_token = layer_to_token,
                                concat_layers = [], control_method = method, head_agg = 'mean',datasize = datasize)
        del data
        torch.cuda.empty_cache()
                     
 
    return
                     
                     
if __name__ == "__main__":
    main(model_name, concept_type)
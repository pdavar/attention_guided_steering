from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import numpy as np
import torch
import matplotlib.pyplot as plt
import direction_utils
from utils import select_llm, ensure_dir, read_file
import os
from tqdm import tqdm
from args import get_args
from datasets import *


ROOT_DIR = "data/attention_to_prompt"

dataset_to_lower = {'fears':True, 
                    'personalities':True, 
                    'moods':True, 
                    'places':False, 
                    'personas':False,
                    'jailbreaking': False}
 

         

def save_attn_paired(llm, concept = 'aggressive', concept_class = 'moods', head_agg = 'mean'):
     
    
    tokenizer = llm.tokenizer
    model = llm.language_model
    model_type = llm.model_name
    NUM_COMMON_TOKS = llm.n_added_tokens
    
    ensure_dir(ROOT_DIR)
    attn_outpath = os.path.join(ROOT_DIR, f"attentions_{head_agg}head_{model_type}_{concept}_paired_statements.npy")
    
    if os.path.exists(attn_outpath): return
    
    paired_dataset_fn = get_dataset_fn(concept_class, paired_samples = True)
    paired_data = paired_dataset_fn(llm, concept) 
    pairs = np.array(paired_data['inputs']) # each element is a tuple (pos, neg)
    pairs = pairs[np.arange(0,len(pairs),2)] #getting a half of the data for the sake of efficiency since our batch size is 1
   
    pos_data = []
    for pos, neg in pairs:
        pos_data.append(pos)
          
     
    n_layers = model.config.num_hidden_layers
    attns = np.zeros((len(pos_data),n_layers,NUM_COMMON_TOKS))
    
    layer_to_attns = direction_utils.get_attns_lastNtoks(pos_data, 
                                                           llm, 
                                                           model,
                                                           llm.tokenizer, 
                                                           NUM_COMMON_TOKS,
                                                          head_agg)
    for layer in range(n_layers):
        attns[:, layer,:] = layer_to_attns[layer]
            
    np.save(attn_outpath, attns)
    print(f"Saved attentions at {ROOT_DIR}")
    return






if __name__=="__main__":
    
    rep_token, model_type, dataset_label, method, _, label = get_args()
    print(f"rep_token = {rep_token}")
    print(f"model_name = {model_type}")
    print(f"concept_type = {dataset_label}")
    print(f"control_method = {method}")
    print(f"labels = {label}")
    assert label in ['hard', 'soft']
    
    llm = select_llm(model_name = model_type, attn_implementation = "eager")
    concept_class = dataset_label
    fname = f"data/concepts/{concept_class}.txt"
    concept_list = read_file(fname, lower=dataset_to_lower[concept_class])
    
        
    for concept in concept_list:
        print(f"=== Concept = {concept} ===")
        
        save_attn_paired(llm, concept , concept_class = concept_class, head_agg = 'mean')
        
    
    
    
    
    
    
    
  
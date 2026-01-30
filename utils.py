import pandas as pd
import numpy as np
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import namedtuple 
from neural_controllers import NeuralController
import re
import os
from collections import Counter
import pickle
from pickle import UnpicklingError
from typing import Dict, Iterable, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from matplotlib.colors import LogNorm
import generation_utils
from tqdm import tqdm
from pathlib import Path


CACHE_DIR = os.environ.get("CACHE_DIR")    #where the models will be downloaded
DATA_DIR = os.path.join(os.getcwd(), "data")    #where the steering_vectors/outputs will be created



SEED = 0
random.seed(SEED)

dataset_to_lower = {'fears':True, 
                    'personalities':True, 
                    'moods':True, 
                    'places':False, 
                    'personas':False}


def get_csv_filename(method, concept_class, rep_token, model_type, version, use_soft_labels):
    root_dir = ensure_dir(os.path.join(DATA_DIR, "csvs"))
    if use_soft_labels:
        filepath = f"{method}_{concept_class}_tokenidx{rep_token}_block_softlabels_gpt4o_outputs_500_concepts_{model_type}_{version}.csv"

    else:
        filepath =  f"{method}_{concept_class}_tokenidx{rep_token}_block_gpt4o_outputs_500_concepts_{model_type}_{version}.csv"
    
    return os.path.join(root_dir, filepath)
        
def get_steered_output_filename(method, concept_class, rep_token, model_type, version, use_soft_labels):
    root_dir = os.path.join(DATA_DIR, 'cached_outputs')
    root_dir = ensure_dir(root_dir)
    if use_soft_labels:
        file_path = f'{method}_{concept_class}_tokenidx{rep_token}_block_softlabels_steered_500_concepts_{model_type}_{version}.pkl'
    else:
        file_path = f'{method}_{concept_class}_tokenidx{rep_token}_block_steered_500_concepts_{model_type}_{version}.pkl'
     
    return os.path.join(root_dir, file_path)
    
def get_concept_vec_filename(method, concept, rep_token, model_type, use_soft_labels):
    root_dir = os.path.join(DATA_DIR, 'directions')
    root_dir = ensure_dir(root_dir)
    if use_soft_labels:
        vec_path = f'{method}_{concept}_tokenidx_{rep_token}_block_softlabels_{model_type}.pkl'
    else:
        vec_path = f'{method}_{concept}_tokenidx_{rep_token}_block_{model_type}.pkl'
   

    return os.path.join(root_dir, vec_path)
    
    

def get_coefs(model_type, use_soft_labels):
    if model_type =="llama_3.3_70b":
        if use_soft_labels:
            return [0.44, 0.46, 0.48, 0.5, 0.52,  0.54, 0.56, 0.58]
        else:
            return [.4, .41, .42, .43, .44, .45]
    elif model_type  =="llama_3.1_70b":
        if use_soft_labels:
            return [0.4,0.42,0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58] 
        else:
            return [.4, .41, .42, .43, .44, .45]
    elif  model_type =="llama_3.1_8b":
        if use_soft_labels:
            return [.55, .6, .65, .7, .75, .8, 0.85,0.9,0.95,1]
        else:
            return [.55, .6, .65, .7, .75, .8]
    elif  model_type =="qwen-14b": 
        if use_soft_labels:
            return [6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5]
        else:
            return [6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5]
    elif  model_type =="qwen-32b": 
        if use_soft_labels:
            return [15,16,17,18,19,20,21]
        else:
            return [15,16,17,18,19,20,21]

def ensure_dir(path):
    """
    Ensure a directory exists; create it (including parents) if it doesn't.
    Returns the directory as a Path.
    """
    p = Path(path)
    print("path is: ", p)
    p.mkdir(parents=True, exist_ok=True)
    return p

"""returns a dictionary {layer: token}"""
def get_tokenidx_per_layer_per_concept(concept, 
                                       model_type, 
                                       metric = 'attn', 
                                       head_agg = 'mean', 
                                       root_dir = "data/attention_to_prompt"):
    assert metric in ['attn','pert']

    outpath = os.path.join(root_dir, f"attentions_{head_agg}head_{model_type}_{concept}_paired_statements.npy")

 
    magnitudes = np.load(outpath).max(axis=0)#(400, 32 layers, n_tokens) --> (32,n_tokens)
    n_tokens = magnitudes.shape[-1]
    max_token_idxs = np.argmax(magnitudes ,axis = 1)-n_tokens #b/c we use negative indexing
    
    return{int(k): int(v) for k, v in zip(np.arange(len(max_token_idxs)), max_token_idxs)}


"""finds what tokens are added at the end by the tokenzier"""
def get_n_common_toks(tokenizer, verbose = False):
    random_word = 'This is a random sentence'
    chat = [{"role": "user", "content": random_word}]

    ids_no_gen = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=False, return_tensors="pt")[0]

    ids_with_gen = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0]

    added_ids = ids_with_gen[len(ids_no_gen)-1:]

    
    n = len(added_ids)
    if verbose: 
        # print("formatted text: ", tokenizer.decode(ids_with_gen))
        print("tokens added at the end:", tokenizer.convert_ids_to_tokens(added_ids.tolist()))
    
    return n

def select_llm(model_name, attn_implementation="eager"):
    MODEL_MAP = {
        # Llama
        "llama_3.1_8b":  "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama_3.1_70b": "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
        "llama_3.3_70b": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",

        
        # Qwen
        "qwen-14b": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        "qwen-32b": "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
    }

    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model_name={model_name!r}. Options: {sorted(MODEL_MAP)}")

    model_id = MODEL_MAP[model_name]


    language_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        cache_dir=CACHE_DIR,
        attn_implementation=attn_implementation,
    ).eval()

    # Keep your logic, but slightly more robust for non-Llama architectures
    archs = getattr(language_model.config, "architectures", []) or []
    use_fast_tokenizer = all("LlamaForCausalLM" not in a for a in archs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=use_fast_tokenizer,
         padding_side="left", 
        legacy=False,
        cache_dir=CACHE_DIR,
    )

    # tokenizer.pad_token_id =  0
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    n = get_n_common_toks(tokenizer, verbose = True)

    LLM = namedtuple("LLM", ["language_model", "tokenizer", "model_name", "n_added_tokens"])
    return LLM(language_model, tokenizer, model_name, n)



def read_file(fname, lower=True):
    concepts = []
    with open(fname, encoding="utf-8") as f: 
        for line in f:
            if lower:
                concepts.append(line.strip().lower())
            else:
                concepts.append(line.strip())
    concepts = sorted(list(set(concepts)))
    return concepts


def compute_save_directions(llm, dataset, use_soft_labels, concept, rep_token,  hidden_state = 'block', 
                            layer_to_token = None, concat_layers = [], control_method='rfm', head_agg = 'mean'):
    
    controller = NeuralController(
            llm,
            llm.tokenizer,
            rfm_iters=8,
            control_method=control_method,
            n_components=1)
    
    if not isinstance(rep_token, int):
        assert layer_to_token is not None
    vec_filename = get_concept_vec_filename(control_method, concept, rep_token, llm.model_name, use_soft_labels)
    try:
        with open(vec_filename, "rb") as f:
            pickle.load(f)
            print("File already exists")
    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
        print("Direction missing or corrupted. Saving direction...")
        controller.compute_directions(dataset['inputs'], dataset['labels'], use_soft_labels,
                                      rep_token, hidden_state, layer_to_token, concat_layers, head_agg)


        controller.save(vec_filename)
        print("Saved steering vector: ", vec_filename)
    return



def select_layers_to_steer(concept, model_type = 'llama_3.1_8b', method = 'topk', k = 3, head_threshold = None, pval_threshold = 0.01):
    """ we are looking for layers where at least one token has significant attention to the prefix """
    x = np.load(f"data/attention_to_prompt/pvalues_{model_type}_{concept}_all_statements.npy")
    layer_to_tok = generation_utils.get_tokenidx_per_layer_per_concept(concept, model_type ,metric = 'attn')
   

    hits = 1.0 * (x < pval_threshold)              # (S, L, H, 4) boolean
        
        # Make an index array that broadcasts over (200, layers, heads)
    toks = np.array(list(layer_to_tok.values()))
    idx_expanded = toks[None, :, None, None]                 # (1, 32, 1, 1)
    idx_expanded = np.broadcast_to(idx_expanded, hits.shape[:-1] + (1,))  # (200, 32, 32, 1)
    hits_in_toks = np.take_along_axis(hits, idx_expanded, axis=-1)[..., 0]  # shape (200, layers, heads)
    hit_avg_over_heads = hits_in_toks.mean(axis=(0,-1)) #(32,)

    if method == 'topk':
        print(f"Choosing top {k} layers to steer.")
        assert k is not None
        layers = np.argsort(hit_avg_over_heads)[::-1][:k]
        values_to_remove = 0
        boolean_mask = layers != values_to_remove
        return layers[boolean_mask]    
    elif method == 'bottomk':
        print(f"Choosing bottom {k} layers to steer.")
        assert k is not None
        layers = np.argsort(hit_avg_over_heads)[:k]
        values_to_remove = 0
        boolean_mask = layers != values_to_remove
        return layers[boolean_mask]
    else:
        print(f"Choosing layers with head-hit average above {head_threshold}.")
        assert head_threshold is not None
        return np.where(hit_avg_over_heads>head_threshold)[0]

    
def generate(concept, llm, prompt, use_soft_labels = True, coefs=[0.4], control_method='rfm', hidden_state = 'block',
             layers_to_control = [],rep_token = -2, max_tokens=100, gen_orig=True, concat_layers_list = [], start_from_token = 'prompt',head_agg = 'max'):
    
    controller = NeuralController(
        llm,
        llm.tokenizer,
        rfm_iters=8,
        control_method=control_method,
        n_components=1,
        start_from_token = start_from_token
    )
    load_concat_layers = len(concat_layers_list)>0
    
    controller.load(concept=concept, 
                    rep_token = rep_token, 
                    model_name=llm.model_name, 
                    path=os.path.join(DATA_DIR,'directions'), 
                    load_concat_layers = load_concat_layers, 
                    hidden_state= hidden_state,
                   use_soft_labels = use_soft_labels,
                   head_agg = head_agg)
    
    

    # No steering 
    if gen_orig:
        original_output = controller.generate(prompt, max_new_tokens=max_tokens,
                                              layers_to_control = [], concat_layers_list = [],
                                              do_sample=False)
        print("Original output:\n",original_output)

    outputs = []
    for coef in coefs:
        if concept == 'jailbreaking': coef*=-1
        print(f"\nCoeff: {coef} ==========================================================")
        steered_output = controller.generate(prompt,
                                            hidden_state = hidden_state,
                                            control_coef=coef,
                                             concat_layers_list = concat_layers_list,
                                            layers_to_control=layers_to_control,
                                            max_new_tokens=max_tokens,
                                            do_sample=False)
        outputs.append((coef, steered_output))
        print(steered_output)
    return outputs


def parse_personality_responses(response, model_type):
    # print(response)
    if model_type == 'llama_3.1_8b' or model_type == 'llama_3.3_70b' or model_type == 'llama_3.1_70b':
        passage = re.split(r"\|>assistant<\|end_header_id\|>", response[1])[1]
    
    elif model_type == 'qwen-14b' or model_type == 'qwen-32b':
        passage = re.split(r"<\|im_start\|>assistant", response[1])[1]
    
    
    passage = "".join(passage)

    return passage


def remove_junk(response):
    # print(response)
    passage = re.split(r"\|>assistant<\|end_header_id\|>", response)[1]
    passage = "".join(passage)

    return passage



def load_prompt(label, version):
    dir = 'data/evaluation_prompts/'
    
    if label == 'fears':
        with open(dir + f'phobia_eval_v{version}.txt', "r") as f:
            return f.read()
    elif label == 'personalities':
        with open(dir + f'personality_eval_v{version}.txt', "r") as f:
            return f.read()
    elif label == 'moods':
        with open(dir + f'mood_eval_v{version}.txt', "r") as f:
            return f.read()
    elif label == 'places':
        with open(dir + f'topophile_eval_v{version}.txt', "r") as f:
            return f.read()
    elif label == 'personas':
        with open(dir + f'persona_eval_v{version}.txt', "r") as f:
            return f.read()
        



def safe_load_pickle(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, UnpicklingError):   # truncated/corrupt
        print(path)
        return None
    except FileNotFoundError:
        print(path)
        return None
    except OSError:
        # e.g., permission issues / short reads on network FS
        print(path)
        return None
    
def validate_output_dict(d: dict, concept_list: list) -> tuple[bool, list[str]]:
    """
    Checks that:
      1) d's keys match concept_list (same set),
      2) each value in d is a list with length >= 6.

    Returns (is_valid, problems).
    """
    problems = []

    keys = set(d.keys())
    target = set(concept_list)

    missing = sorted(target - keys)
    extra   = sorted(keys - target)
    if missing:
        problems.append(f"Missing keys: {missing}")
    if extra:
        problems.append(f"Unexpected keys: {extra}")

    for k in keys & target:
        v = d[k]
        if not isinstance(v, list):
            problems.append(f"{k}: value is {type(v).__name__}, expected list")
        elif len(v) < 6:
            problems.append(f"{k}: list length {len(v)} (< 6)")

    return (len(problems) == 0, problems)
    
    




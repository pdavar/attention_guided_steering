import torch
import random
import numpy as np
import generation_utils
import direction_utils
from control_toolkits import *
import os
import pickle
from tqdm import tqdm
import shutil
import utils
import pandas as pd


SEED = 0
random.seed(SEED)               
np.random.seed(SEED)            
torch.manual_seed(SEED)         
torch.cuda.manual_seed(SEED) 

TOOLKITS = {
    'rfm' : RFMToolkit,
    'linear' : LinearProbeToolkit,
    'logistic' : LogisticRegressionToolkit,
    'mean_difference' : MeanDifferenceToolkit,
    'pca' : PCAToolkit
}


class NeuralController:
    def __init__(self, llm, tokenizer, control_method='rfm', n_components=5, 
                 rfm_iters=8, batch_size=16, start_from_token = 0):
        self.llm = llm
        self.model = llm.language_model.eval()

        self.language_model = self.model
            
        self.tokenizer = tokenizer
        self.control_method = control_method
        self.model_name = llm.model_name
        self.concat_directions = None
        self.hyperparams = {
            'control_method' : control_method,
            'rfm_iters' : rfm_iters,
            'forward_batch_size' : batch_size,
            'M_batch_size' : 2048,
            'n_components' : n_components,
        }
        
        if type(start_from_token) == int:
            self.start_from_token = start_from_token
        else:
            self.start_from_token = generation_utils.get_userprompt_start(self.tokenizer) #used for the hook
        print(f"steering starting from token index {self.start_from_token}")
        self.toolkit = TOOLKITS[control_method]()
        
        print("\nController hyperparameters:")
        for n_, v_ in self.hyperparams.items():
            print(f"{n_:<20} : {v_}")


   
        
    def compute_directions(self, data, labels, use_soft_labels, rep_token, hidden_state = 'block', layer_to_token = None, concat_layers = [], head_agg = 'mean'):
        hidden_layers = list(range(1, self.model.config.num_hidden_layers, 1))
        
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels).reshape(-1,1)
        self.directions, self.rs, _, _ = self.toolkit._compute_directions(data, 
                                                       labels,
                                                       use_soft_labels,
                                                       self.llm,
                                                       self.model, 
                                                       self.tokenizer,     
                                                       hidden_layers,
                                                       rep_token, 
                                                        layer_to_token,
                                                       self.hyperparams,
                                                        head_agg)
          
    def compute_concat_directions(self, data, labels, rep_token, hidden_state, concat_layers):
        assert len(concat_layers)>0
        self.concat_directions = {}
        for layer_pair_list in concat_layers:
            u, M = self.toolkit._compute_AGOP_concat(data, 
                                                           labels,
                                                           self.llm,
                                                           self.model, 
                                                           self.tokenizer, 
                                                           layer_pair_list,
                                                           rep_token,
                                                           hidden_state,
                                                           self.hyperparams,
                                                          )
            self.concat_directions[tuple(layer_pair_list)] = u
        return
 
    def save(self, filename):

        with open(filename, 'wb') as f:
            pickle.dump(self.directions, f)
        return
    
    
    
    def save_only_if_different(self, concept, model_name, path='./'):
        filename = os.path.join(path, f'{self.control_method}_{concept}_{model_name}.pkl')
         
        with open(filename, 'rb') as f:
            old_directions = pickle.load(f)
        
        all_close = all(torch.allclose(old_directions[k], self.directions[k],\
                                     rtol=1e-1) for k in old_directions.keys())
        
        if all_close:
            print("keeping old direction")
        else:
            print("Saving new direction")
            print(self.directions[1], old_directions[1])
            with open(filename, 'wb') as f:
                pickle.dump(self.directions, f)
            with open(f"new_computations{model_name}.txt", "a") as f:
                f.write("\n"+filename)
    
        return

            
    def load(self, concept, rep_token, model_name, path='./', load_concat_layers = False, hidden_state = 'block', use_soft_labels = True, head_agg = 'max'):
        print("inside load: ", rep_token)
        if isinstance(rep_token, int):
            print(f"All layers use the same token index {rep_token}")
            layer_to_tok = None
            
            filename = utils.get_concept_vec_filename(self.control_method, concept, rep_token, model_name, use_soft_labels)
            print("loading this file: ", filename)
            with open(filename, 'rb') as f:
                self.individual_directions = pickle.load(f)
       
    
        else:
            assert rep_token == 'max_attn_per_layer'
            filename = utils.get_concept_vec_filename(self.control_method, concept, rep_token, model_name, use_soft_labels)
            if os.path.exists(filename):
                print("loading this file: ", filename)
                with open(filename, 'rb') as f:
                    self.individual_directions = pickle.load(f)
                
            else:
                raise FileNotFoundError(f"File not found: {filename}")

                # layer_to_tok = generation_utils.get_tokenidx_per_layer_per_concept(concept, model_name, metric = 'attn', head_agg = head_agg)
                # print(f"using rep token {rep_token}: {layer_to_tok}")
                # self.individual_directions = {}
                # non_nan_layers = [k for k, v in layer_to_tok.items() if not pd.isna(v)]
                # non_nan_layers.pop(0) # we don't steer the first layer

                # for layer in non_nan_layers:
                #     tok_idx = int(layer_to_tok[layer])
                #     assert tok_idx in [-1,-2,-3,-4, -5]
                #     filename = utils.get_concept_vec_filename(self.control_method, concept, tok_idx, model_name, use_soft_labels)
                #     print("loading this file: ", filename)
                #     self.individual_directions[layer] = pickle.load(open(filename, 'rb'))[layer]

        
        return
        
    def format_prompt(self, prompt, role='user'):
        chat = [{"role": "user", "content": prompt}]
        out = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return out
    
     # # Assumes non formatted prompt
    def generate(self, plaintext_prompt, hidden_state = 'block', control_coef=0.4,layers_to_control=[],concat_layers_list=[],  **kwargs):        
        prompt = self.format_prompt(plaintext_prompt)
        if len(layers_to_control) == 0:
            return generation_utils.generate_on_text(self.model, self.tokenizer, prompt, **kwargs)
        else:
            print("generating with perturbing these layers: ", layers_to_control)
            self.get_all_directions(concat_layers_list) #replaces individual directions with concatenated 
            return self._controlled_generate(prompt, layers_to_control, control_coef,hidden_state, **kwargs)


    def _controlled_generate(self, prompt, layers_to_control, control_coef, hidden_state, **kwargs):
        ## define hooks
        hooks = generation_utils.hook_model(self.model, self.directions, layers_to_control,
                                            control_coef, self.start_from_token)

        ## do forward pass
        out = generation_utils.generate_on_text(self.model, self.tokenizer, prompt, **kwargs)

        ## clear hooks
        generation_utils.clear_hooks(hooks)
        return out

   
    def get_all_directions(self, concat_layers_list):
        if self.concat_directions is None or len(concat_layers_list)==0:
            self.directions = self.individual_directions.copy()
            print("using all individual directions")
        else:
            self.directions = {}
            assert len(concat_layers_list)==2, print(concat_layers_list)
            print("using concatenated directions: ",concat_layers_list)
            #first, populate the directions witch those of the concatenated directions
            l1, l2 = concat_layers_list[0],concat_layers_list[1]
            assert (l1,l2) in self.concat_directions.keys()
            combined_direction = self.concat_directions[(l1,l2)]
            self.directions[l1] = combined_direction[:,:4096] / combined_direction[:,:4096].norm(p=2)
            self.directions[l2] = combined_direction[:,4096:] / combined_direction[:,4096:].norm(p=2)
            for individual_layer_idx in self.individual_directions.keys():
                if individual_layer_idx not in self.directions.keys():
                    self.directions[individual_layer_idx] = self.individual_directions[individual_layer_idx]
        return 

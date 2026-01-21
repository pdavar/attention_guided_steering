import torch
import pandas as pd
import numpy as np
import pickle
import types
import math 
import torch.nn.functional as F
import os




def generate_on_text(model, tokenizer, input_text, **kwargs):
        
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(model.device)

    stop_token_id = getattr(model.config, "eos_token_id", None) or tokenizer.eos_token_id

    # stop_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # Generate output
    outputs = model.generate(
        **inputs,
        eos_token_id=stop_token_id,        
        **kwargs)
    
    # Decode the output
    generated_text = tokenizer.decode(outputs[0])
    return generated_text

def generate_on_text_with_attn(model, tokenizer, input_text, **kwargs):
        
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(model.device)

    stop_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # Generate output
    outputs = model.generate(
        **inputs,
        eos_token_id=stop_token_id,        
        **kwargs)
    

    # Decode the output
    generated_text = tokenizer.decode(outputs[0])
    return generated_text





        
# def hook_model(model, directions, layers_to_control, control_coef, hidden_state = 'block'):
    
    
#     # For multimodal models, hook only the language model layers
#     if hasattr(model, 'language_model'):
#         layers = model.language_model.model.layers
#     else:
#         layers = model.model.layers
        
        
#     if hidden_state=='block':
#         hooks = {}
#         for layer_idx in layers_to_control:

#             control_vec = directions[layer_idx]
#             if len(control_vec.shape)==1:
#                 control_vec = control_vec.reshape(1,1,-1)
            
#             block = layers[layer_idx]

#             def block_hook(module, input, output, control_vec=control_vec, control_coef=control_coef):
#                 new_output = output[0] if isinstance(output, tuple) else output
#                 new_output = new_output + control_coef*control_vec.to(dtype=new_output.dtype, device=new_output.device)

#                 if isinstance(output, tuple):
#                     new_output = (new_output,) + output[1:] 

#                 return new_output

#             hook_handle = block.register_forward_hook(block_hook)
#             hooks[layer_idx] = hook_handle
#         return hooks
    
#     elif hidden_state == 'attn':
#         hooks = []

#         for layer_idx in layers_to_control:
#             control_vec = directions[layer_idx]
#             if len(control_vec.shape) == 1:
#                 control_vec = control_vec.reshape(1, 1, -1)

#             ln_module = model.model.layers[layer_idx].post_attention_layernorm  # LayerNorm after attention

#             def make_hook(control_vec=control_vec, control_coef=control_coef):
#                 def ln_hook(module, input, output):
#                     x = output + control_coef * control_vec.to(dtype=output.dtype, device=output.device)
#                     return x
#                 return ln_hook

#             hook = ln_module.register_forward_hook(make_hook())
#             hooks.append(hook)
#         return hooks

#     else:
#         raise ValueError
   
    
def get_userprompt_start(tokenizer):
    plaintext_prompt = "hi how are you doing?"

    chat = [{"role": "user", "content": plaintext_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    tokenized_prompt = tokenizer(formatted_prompt, return_tensors='pt',add_special_tokens=False)['input_ids']
    user_start_id = tokenizer.convert_tokens_to_ids('hi')
    idx = torch.where(tokenized_prompt==user_start_id)[1][0] 
    return idx

# def hook_model(model, directions, layers_to_control, control_coef, start_from_token):
#     """
#     Adds `control_vec` only to positions >= start_from_token in each selected layer's output.
#     """
#     # For multimodal models, hook only the language model layers
#     if hasattr(model, 'language_model'):
#         layers = model.language_model.model.layers
#     else:
#         layers = model.model.layers

#     hooks = {}
#     for layer_idx in layers_to_control:
#         control_vec = directions[layer_idx]
#         if control_vec.ndim == 1:
#             control_vec = control_vec.reshape(1, 1, -1)  # (1,1,D) for broadcasting

#         block = layers[layer_idx]

#         def block_hook(module, input, output,
#                        control_vec=control_vec, control_coef=control_coef, start_from_token=start_from_token):
#             # Normalize to a Tensor (some models return tuples)
#             new_output = output[0] if isinstance(output, tuple) else output
#             B, T, D = new_output.shape

#             # mask positions >= start_from_token
#             pos_mask = (torch.arange(T, device=new_output.device)[None, :, None] >= int(start_from_token))
#             pos_mask = pos_mask.to(new_output.dtype)  # broadcast to (1,T,1)

#             delta = control_coef * control_vec.to(dtype=new_output.dtype, device=new_output.device)
#             new_output = new_output + delta * pos_mask  # (B,T,D) + (1,1,D) * (1,T,1)

#             if isinstance(output, tuple):
#                 return (new_output,) + output[1:]
#             return new_output

#         hook_handle = block.register_forward_hook(block_hook)
#         hooks[layer_idx] = hook_handle
#     return hooks


def hook_model(model, directions, layers_to_control, control_coef, start_from_token):
    """
    Register forward hooks that add `control_vec` to each selected transformer block's *output*
    for token positions with absolute index >= `start_from_token`.

    Args:
        model: HF-style model (optionally multimodal with `language_model` attribute).
        directions: dict or list mapping layer_idx -> control_vec (D,) or (1,1,D) tensor/ndarray.
        layers_to_control: iterable of layer indices to hook.
        control_coef: float or scalar tensor multiplier for control_vec.
        start_from_token: int absolute token index threshold (0-based).

    Returns:
        dict[layer_idx] -> hook handle (so you can later .remove()).
    """
    # For multimodal wrappers, only hook the language model blocks
    layers = getattr(getattr(model, 'language_model', model).model, 'layers')

    hooks = {}
    for layer_idx in layers_to_control:
        control_vec = directions[layer_idx]
        if not torch.is_tensor(control_vec):
            control_vec = torch.tensor(control_vec)
        if control_vec.ndim == 1:
            control_vec = control_vec.view(1, 1, -1)  # (1,1,D) for clean broadcasting

        block = layers[layer_idx]

        def block_hook(module, inputs, output,
                       control_vec=control_vec,
                       control_coef=control_coef,
                       start_from_token=start_from_token):
            """
            inputs is typically: (hidden_states, attention_mask, position_ids, *extras)
            output may be a tensor or a tuple with hidden_states first.
            """
            # Normalize output to a tensor
            new_hidden = output[0] if isinstance(output, tuple) else output
            B, T, D = new_hidden.shape

            # Try absolute positions from position_ids; fallback to attention_mask-derived offset
            abs_pos = None
            if len(inputs) >= 3 and inputs[2] is not None:
                # position_ids: (B, T) absolute positions
                abs_pos = inputs[2]
            else:
                # attention_mask: (B, total_seq_len_seen)
                past_len = 0
                if len(inputs) >= 2 and inputs[1] is not None:
                    total_seen = inputs[1].shape[-1]
                    past_len = int(total_seen - T)
                # Build absolute positions for this chunk
                abs_pos = torch.arange(past_len, past_len + T, device=new_hidden.device).view(1, T).expand(B, T)

            # Mask for absolute positions >= start_from_token -> shape (B, T, 1)
            pos_mask = (abs_pos >= int(start_from_token)).unsqueeze(-1).to(dtype=new_hidden.dtype)

            # Prepare delta (broadcast control vec and coef)
            delta = control_coef * control_vec.to(dtype=new_hidden.dtype, device=new_hidden.device)  # (1,1,D)

            # Apply perturbation
            new_hidden = new_hidden + delta * pos_mask  # (B,T,D) + (B,T,1)*(1,1,D)
            # Restore original output structure
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        hooks[layer_idx] = block.register_forward_hook(block_hook)

    return hooks


def clear_hooks(hooks) -> None:
    if isinstance(hooks, list):
        for hook_handle in hooks:
            hook_handle.remove() 
    else:
        for hook_handle in hooks.values():
            hook_handle.remove()
        

    
# def get_layer_to_token_mapping(rep_token, concept, model_type, pval_threshold = 0.01):
#     if isinstance(rep_token, int):
#         print("using integer rep token for all layers: ", rep_token)
#         return None
#     else:
#         assert rep_token in [ 'max_attn_per_layer', 'max_pert_per_layer']
#         if rep_token == 'max_per_layer_avgstmnt':
#             return get_tokenidx_per_layer_per_concept(concept, model_type, statement_agg_method = 'mean')
#         elif rep_token == 'max_per_layer_maxstmnt':
#             return get_tokenidx_per_layer_per_concept(concept, model_type, statement_agg_method = 'max')
        
#         x = np.load(f"data/attention_to_prompt/pvalues_{model_type}_{concept}_all_statements.npy") # (statements,layers, heads, toks)
#         hits = 1.0*(x<pval_threshold)
#         if rep_token == 'max_hit_per_layer_maxstmnt':
#             hits = hits.sum(axis = 2).max(axis = 0)
#         else:
#             hits = hits.sum(axis = 2).mean(axis = 0) #sum over heads, then average over statements. final shape is (layers, toks)
        
#         max_idx = hits.argmax(axis=1)
#         mx  = hits[np.arange(hits.shape[0]), max_idx]
#         out = max_idx.astype(float)
        
#         if rep_token == 'max_hit_per_layer' or rep_token == 'max_hit_per_layer_maxstmnt':
#             # 1) Modified: map layer -> tok, but NaN if all token hit_sums in that layer are 0
#             out[mx == 0] = np.nan
            
#             return {i:out[i]-4 for i in range(len(hits))}

#         elif rep_token == 'max_onehit_per_layer':

#             out[mx <= 1] = np.nan
            
#             return {i:out[i]-4 for i in range(len(hits))}
    
#         elif rep_token == 'max_twohit_per_layer':

#             out[mx <= 2] = np.nan
            
#             return {i:out[i]-4 for i in range(len(hits))}
        
#         elif rep_token == 'max_fivehit_per_layer':

#             out[mx <= 5] = np.nan
            
#             return {i:out[i]-4 for i in range(len(hits))}
        


        

 
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import rfm
import pickle
import numpy as np
from copy import deepcopy
from torchmetrics.regression import R2Score




def get_prefix_attn_sum_for_layer_lastN(attn_for_layer, # (batch, num_heads, prompt_len (from), prompt_len(to))
                                          N,
                                          prefix_start, 
                                          prefix_end,
                                            head_agg = 'mean'): 
    assert attn_for_layer.shape[0]==1, "batch size must be 1"
    n_heads = attn_for_layer.shape[1]
    
    head_attns = []
    for h in range(n_heads):
        head_attn_to_prefix = attn_for_layer[0, h, -N:, prefix_start:prefix_end].sum(-1)  # (N,)
        head_attns.append(head_attn_to_prefix.cpu().numpy())  
    if head_agg == 'mean':
        return [np.mean(head_attns, axis = 0)] #(N)
    elif head_agg == 'max':
        return [np.max(head_attns, axis = 0)] #(N)
    else:
        raise ValueError
        
"""returns a list of length batch_size containing the prefix attention sums averaged over heads"""
def get_prefix_attn_sum_for_layer_singletoken(attn_for_layer, # (batch, num_heads, prompt_len (from), prompt_len(to))
                                  rep_token,
                                  prefix_start, 
                                  prefix_end,
                                    head_agg = 'mean'): 
    assert attn_for_layer.shape[0]==1, "batch size must be 1"
    n_heads = attn_for_layer.shape[1]
    
    head_attns = []
    for h in range(n_heads):
        head_attn_to_prefix = attn_for_layer[0, h, rep_token, prefix_start:prefix_end].sum(-1)  # (n_tokens,)
        # head_attns.append(head_attn_to_prefix.cpu().numpy())  
        head_attns.append(head_attn_to_prefix.detach().to(torch.float16).cpu().numpy())

    if head_agg == 'mean':
        return [np.mean(head_attns, axis = 0)] #(n_tokens)
    elif head_agg == 'max':
        return [np.max(head_attns, axis = 0)] #(n_tokens)
    else:
        raise ValueError


def get_hidden_states_and_attns(prompts, labels, llm, model, tokenizer,
                                 hidden_layers, rep_token, layer_to_token, head_agg):
    
    pos_inds = np.where(np.array(labels)==1)[0]
    pos_prompts = np.array(prompts)[pos_inds]

    print("prompt 1: ", prompts[0])
    print("prompt 2: ", prompts[1])
    #these indices are shared across all statements
    prefix_start, prefix_end = get_prefix_inds(pos_prompts[0], tokenizer) #these are positive indices (indexing from the start)
    
   

    # Initialize storage with proper indices
    all_hidden_states = {layer_idx:[] for layer_idx in hidden_layers}
    soft_labels = {layer_idx:[] for layer_idx in hidden_layers}# 0 for negative examples, prefix_attn for positive samples
   
    with torch.no_grad():
        for i,prompt in tqdm(enumerate(prompts)):
         
            encoded_inputs = tokenizer(prompt, return_tensors='pt',
                                       padding=False, 
                                       add_special_tokens=False).to(model.device)

            input_ids, attention_mask = encoded_inputs['input_ids'], encoded_inputs['attention_mask'].half()
            
            
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True)
           
            """
            batch_attentions has length num_layers
            batch_attentions[0] has shape (batch, num_heads, prompt_len (from), prompt_len(to))
            """
            # print("my input ids: ", input_ids)
            out_hidden_states = outputs.hidden_states 
            # neg_batch_attentions, neg_out_hidden_states = neg_outputs.attentions, neg_outputs.hidden_states 

            hidden_states_list = list(out_hidden_states)[1:]# Skip embedding layer
            
            
            
        #out_hidden_states_list is a list of each layer's hidden state with dim (batch, seq_len,4096)
            for layer_idx in range(len(hidden_states_list)):
                
                if not isinstance(rep_token, int): 
                    layer_rep_token = layer_to_token[layer_idx]
                else:
                    layer_rep_token = rep_token
                
                
                if layer_idx in hidden_layers:
                    
                    layer_hidden_states = hidden_states_list[layer_idx][0,layer_rep_token,:].detach().cpu()
                    all_hidden_states[layer_idx].append(layer_hidden_states)
                    
                    # print("emb shape: ", all_hidden_states[layer_idx].shape)
                    if labels[i]==1:
                        layer_attns = outputs.attentions[layer_idx]
                       
                        attns_all_toks =  get_prefix_attn_sum_for_layer_singletoken(layer_attns, 
                                                                                  layer_rep_token,
                                                                                  prefix_start, 
                                                                                  prefix_end,
                                                                                   head_agg)
                        
                        
                        soft_labels[layer_idx]+=attns_all_toks #(n_tokens_after_prefix)
                    else:
                        soft_labels[layer_idx]+=[0]
                    assert len(all_hidden_states[layer_idx]) == len(soft_labels[layer_idx]), f"{len(all_hidden_states[layer_idx])}, {len(soft_labels[layer_idx])}, {soft_labels[layer_idx]}"
  
    final_hidden_states = {}
    for layer_idx in hidden_layers:
        final_hidden_states[layer_idx] = torch.stack(all_hidden_states[layer_idx], dim=0)
        soft_labels[layer_idx] = torch.tensor(soft_labels[layer_idx]).unsqueeze_(1)    
        
    
    return final_hidden_states , soft_labels    #shapes: torch.Size([400xnum_tokens, 4096]), torch.Size([400xnum_tokens,1])




def get_attns_lastNtoks(pos_prompts, llm, model, 
                    tokenizer, num_common_toks, head_agg):
    
    n_layers = model.config.num_hidden_layers
    print("pos prompt 1: ", pos_prompts[0])
    #these indices are shared across all statements
    prefix_start, prefix_end = get_prefix_inds(pos_prompts[0], tokenizer) #these are positive indices (indexing from the start)
    
   

    # Initialize storage with proper indices
    attns = {k:[] for k in range(n_layers)}# 0 for negative examples, prefix_attn for positive samples
   
    with torch.no_grad():
        for i,prompt in tqdm(enumerate(pos_prompts)):
         
            encoded_inputs = tokenizer(prompt, return_tensors='pt',
                                       padding=False, 
                                       add_special_tokens=False).to(model.device)

            input_ids, attention_mask = encoded_inputs['input_ids'], encoded_inputs['attention_mask'].half()
            
            if i==0: print(f"\ndecoded prompt: [{llm.tokenizer.decode(input_ids[0])}]")
            if i==0: print(f"common tokens: [{llm.tokenizer.convert_ids_to_tokens(input_ids[0,-num_common_toks:].tolist())}]")
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=False,
                output_attentions=True,
                return_dict=True)
            assert len(outputs.attentions)==n_layers
           
            """
            batch_attentions has length num_layers
            batch_attentions[0] has shape (batch, num_heads, prompt_len (from), prompt_len(to))
            """
        #out_hidden_states_list is a list of each layer's hidden state with dim (batch, seq_len,4096)
            for layer_idx in range(n_layers):
    
                layer_attns = outputs.attentions[layer_idx]

                attns_all_toks =  get_prefix_attn_sum_for_layer_lastN(layer_attns, 
                                                                          num_common_toks,
                                                                          prefix_start, 
                                                                          prefix_end,
                                                                           head_agg)


                attns[layer_idx]+=attns_all_toks #(num_common_toks)
                    
  
    for layer_idx in range(n_layers):
        attns[layer_idx] = torch.tensor(attns[layer_idx])  
        
    
    print("look here: ", attns[1].shape)
    return  attns    #shapes: torch.Size([400xnum_tokens, 4096]), torch.Size([400xnum_tokens,1])

def get_n_prepend_toks(tokenizer, verbose=False, text="this is a random sentence"):
    chat_full  = [{"role": "user", "content": text}]
    chat_empty = [{"role": "user", "content": ""}]

    # Tokenize chat template directly (fast and tokenizer-agnostic)
    ids_full = tokenizer.apply_chat_template(
        chat_full,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    ids_empty = tokenizer.apply_chat_template(
        chat_empty,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )

    # Make sure we have 1D tensors
    ids_full = ids_full[0] if isinstance(ids_full, torch.Tensor) and ids_full.ndim == 2 else ids_full
    ids_empty = ids_empty[0] if isinstance(ids_empty, torch.Tensor) and ids_empty.ndim == 2 else ids_empty

    # Longest common prefix length = number of tokens before the user text starts
    L = min(len(ids_full), len(ids_empty))
    i = 0
    while i < L and ids_full[i].item() == ids_empty[i].item():
        i += 1

    n_prefix = i

    if verbose:
        # Show the prefix tokens (what got added before content)
        prefix_ids = ids_full[:n_prefix]
        print("formatted text:", tokenizer.apply_chat_template(chat_full, tokenize=False, add_generation_prompt=False))
        print("n_prepended:", n_prefix)
        print("prefix tokens:", tokenizer.convert_ids_to_tokens(prefix_ids.tolist()))
        print("pre-prefix decoded:", tokenizer.decode(prefix_ids.tolist(), skip_special_tokens=False))

    return n_prefix


def get_prefix_inds(pos_prompt, tokenizer):
    encoded_prompt = tokenizer(pos_prompt, return_tensors='pt', padding=False, add_special_tokens=False)['input_ids']
    input_ids = encoded_prompt[0]  # shape: (seq_len,)

        
    personify_idx = get_n_prepend_toks(tokenizer, verbose = False)
    
    what_idx = torch.where(input_ids==tokenizer.encode(" What", add_special_tokens=False)[0])[0]
    if len(what_idx)>1: what_idx = what_idx[0]
    what_idx = what_idx.item()

    # print(personify_idx, what_idx,encoded_prompt.shape, encoded_prompt)
    # print("look here: ", tokenizer.decode(encoded_prompt[0,:personify_idx]))
    print(f"\n==== prefix: [{tokenizer.decode(encoded_prompt[0,personify_idx:what_idx])}]", )
    
    return (personify_idx, what_idx) 


def train_rfm_probe_on_concept(train_X, train_y, 
                               val_X, val_y, 
                               bws=[1, 10, 100],
                               reg = 1e-3):
    
   
    best_r = -float('inf')
    best_M = None
    train_X = train_X.cuda()
    train_y = train_y.cuda()
    val_X = val_X.cuda()
    val_y = val_y.cuda()


    for bw in bws:
        for norm in [True, False]:
        
            u,r, M = adit_rfm.rfm((train_X, train_y), (val_X, val_y), L=bw, 
                                    reg=reg, num_iters=10, norm=norm)
            # print("stats: ", bw, norm, r)
            if r >= best_r: 
                best_u = u 
                best_r = r
                best_M = M
                best_bw = bw
                best_norm = norm
              

            torch.cuda.empty_cache()

 
    print(f'Best RFM r: {best_r}, bw: {best_bw}, norm: {best_norm}')


    return best_u, best_r, best_M

def project_onto_direction(tensors, direction, device='cuda'):
    """
    tensors : (n, d)
    direction : (d, )
    output : (n, )
    """
    # print("tensors", tensors.shape, "direction", direction.shape)
    assert(len(tensors.shape)==2)
    assert(tensors.shape[1] == direction.shape[0])
    
    return tensors.to(device=device) @ direction.to(device=device, dtype=tensors.dtype)


def pearson_corr(x, y):     
    assert(x.shape == y.shape)
    
    x = x.float() + 0.0
    y = y.float() + 0.0

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    numerator = torch.sum(x_centered * y_centered)
    denominator = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2))

    return numerator / denominator


def fit_pca_model(train_X, train_y, val_X, val_y, n_components=1):
    """
    Assumes the data are in ordered pairs of pos/neg versions of the same prompts:
    
    e.g. the first four elements of train_X correspond to 
    
    Dishonestly say something about {object x}
    Honestly say something about {object x}
    
    Honestly say something about {object y}
    Dishonestly say something about {object y}
    

    """
    assert ((train_y == 0) | (train_y == 1)).all(), "Training labels must be 0 or 1"

    pos_indices = torch.isclose(train_y, torch.ones_like(train_y)).squeeze(1)
    neg_indices = torch.isclose(train_y, torch.zeros_like(train_y)).squeeze(1)
    
    pos_examples = train_X[pos_indices]
    neg_examples = train_X[neg_indices]
    
    dif_vectors = pos_examples - neg_examples
    
    # randomly flip the sign of the vectors
    random_signs = torch.randint(0, 2, (len(dif_vectors),)).float().to(dif_vectors.device) * 2 - 1
    dif_vectors = dif_vectors * random_signs.reshape(-1,1)
    
    # dif_vectors : (n//2, d)
    XtX = dif_vectors.T@dif_vectors
    s, u = torch.lobpcg(XtX, k=1)

    preds = val_X @ u     
    # print(preds.shape, y.shape)
    test_r = torch.abs(torch.corrcoef(torch.cat((preds, val_y), dim=-1).T))[0, 1].item()
    print("Test r: ", test_r)


    return u.reshape(1, -1)


def train_linear_probe_on_concept(train_X, train_y, val_X, val_y, use_bias=False):
    
    if use_bias:
        X = append_one(train_X)
        Xval = append_one(val_X)
    else:
        X = train_X
        Xval = val_X
    
    n, d = X.shape
    num_classes = train_y.shape[1]

    best_loss = float('inf')
    best_beta = None
    for reg in [1e-4, 1e-3, 1e-2,1e-1, 1, 10]:

        X = X.cuda()
        train_y = train_y.cuda()
        XtX = X.T @ X
        Xty = X.T @ train_y
        try:
            beta = torch.linalg.solve(XtX + reg*torch.eye(X.shape[1]).cuda(), Xty) 
        except:
            print("reg: ", reg)
            raise ValueError
        
        
        preds = Xval.cuda() @ beta

        val_loss = torch.mean((preds-val_y.cuda())**2)

        if preds.shape[1] == 1:
            r2score = R2Score().cuda()
            val_r2 = r2score(preds, val_y.cuda()).item()
        else:
            val_r2 = None

        if val_loss < best_loss:
            best_val_r2 = val_r2
            best_loss = val_loss
            best_reg = reg
            best_beta = deepcopy(beta)
            best_acc = accuracy_fn(preds, val_y)
    
    print(f'Linear probe loss: {best_loss}, R2: {best_val_r2}, reg: {best_reg}, acc: {best_acc}')

    if use_bias:
        line = best_beta[:-1].to(train_X.device)
        if num_classes == 1:
            bias = best_beta[-1].item()
        else:
            bias = best_beta[-1]
    else:
        line = best_beta.to(train_X.device)
        bias = 0
        
    return line, bias

def accuracy_fn(preds, truth):
    assert(len(preds)==len(truth))
    true_shape = truth.shape
    
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds).to(truth.device)
    preds = preds.reshape(true_shape)
    
    if preds.shape[1] == 1:
        preds = torch.where(preds >= 0.5, 1, 0)
        truth = torch.where(truth >= 0.5, 1, 0)
    else:
        preds = torch.argmax(preds, dim=1)
        truth = torch.argmax(truth, dim=1)
        
    acc = torch.sum(preds==truth)/len(preds) * 100
    return acc.item()
import torch
from sklearn.linear_model import LogisticRegression
import numpy as np
import direction_utils
from sklearn.metrics import log_loss
import pickle

from tqdm import tqdm
import time
from copy import deepcopy

def split_indices(N, frac=0.2, max_val_count=256, random_split=False):
    n_train = N - min(int(frac*N), max_val_count)
    n_train = n_train + n_train%2 # ensure even train samples
    
    if random_split:
        indices = list(range(N))
        random.shuffle(indices)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
    else:
        train_indices = range(n_train)
        val_indices = range(n_train, N)
    return train_indices, val_indices

def minmax_normalize(y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Min-max normalize a tensor to [0, 1].

    - Works on any shape.
    - NaNs are ignored for min/max (if present) and remain NaN in the output.
    - If all (non-NaN) values are constant, returns zeros (or NaNs where input is NaN).
    """

    y_min = torch.min(y)
    y_max = torch.max(y)
    denom = y_max - y_min

    # If denom is ~0 (constant), avoid divide-by-zero
    out = (y - y_min) / denom 
    out = torch.where(denom.abs() < eps, torch.zeros_like(out), out)
    return out


class RFMToolkit():
    def __init__(self):
        pass

    def _compute_directions(self, data, labels, use_soft_labels, 
                            llm, model, tokenizer, hidden_layers, 
                            rep_token, layer_to_token, hyperparams, head_agg):
        
                  
        print("Using soft labels: ", use_soft_labels)
        n_components = hyperparams['n_components']
        train_indices, val_indices = split_indices(len(data))
        num_classes = labels.shape[1]
        
       
        hidden_states, layer_to_soft_labels = direction_utils.get_hidden_states_and_attns(data, labels,llm, model,
                                                                                  tokenizer, hidden_layers, 
                                                                                          rep_token,layer_to_token, head_agg)
                                                       
        # print("My hidden states: ", hidden_states.keys(), hidden_states[39],  hidden_states[1])
            
        assert hidden_states[hidden_layers[0]].shape[0] == layer_to_soft_labels[hidden_layers[0]].shape[0]
      
            
                
            
            
        train_indices, val_indices = split_indices(len(hidden_states[hidden_layers[0]]))
        print("shape of hidden states: ", hidden_states[hidden_layers[0]].shape)
        
        
        
        directions = {}
        rs = {}
        signs = {}

        for layer_to_eval in tqdm(hidden_layers):
            hidden_states_at_layer = hidden_states[layer_to_eval].cuda().float()
            all_y = layer_to_soft_labels[layer_to_eval].float().cuda() if use_soft_labels else labels.float().cuda()
            
            hidden_states_at_layer = hidden_states_at_layer/torch.linalg.norm(hidden_states_at_layer, dim = 1, keepdim = True)###############
            all_y = minmax_normalize(all_y)#############################################

            train_y = all_y[train_indices]
            val_y = all_y[val_indices]
            
            
            train_X = hidden_states_at_layer[train_indices] 
            val_X = hidden_states_at_layer[val_indices]
                
            if layer_to_eval==2:
                print("train X shape:", train_X.shape, "train y shape:", train_y.shape, 
                      "val X shape:", val_X.shape, "val y shape:", val_y.shape)
                print("val_y: ", val_y)
            assert(len(train_X) == len(train_y))
            assert(len(val_X) == len(val_y))
            
            # print(train_X, train_y)
            
            
            
            # if layer_to_eval==31: print("look here: ")
            u, r, M = direction_utils.train_rfm_probe_on_concept(train_X, train_y, val_X, val_y)

            directions[layer_to_eval] = u.reshape(1, -1)
            rs[layer_to_eval] = r
                
        signs = {}
        if num_classes == 1: # only if binary do you compute signs
            signs = self._compute_signs(hidden_states, all_y, directions, n_components)
            for layer_to_eval in tqdm(hidden_layers):
                for c_idx in range(n_components):
                    directions[layer_to_eval][c_idx] *= signs[layer_to_eval][c_idx]
        return directions, rs, hidden_states, layer_to_soft_labels

    def _compute_signs(self, hidden_states, all_y, directions, n_components):
        
        signs = {}
        for layer in hidden_states.keys():
            xs = hidden_states[layer]
            signs[layer] = {}
            for c_idx in range(n_components):
                direction = directions[layer][c_idx]
                hidden_state_projections = direction_utils.project_onto_direction(xs, direction).to(all_y.device)
                sign = 2*(direction_utils.pearson_corr(all_y.squeeze(1), hidden_state_projections) > 0) - 1
                signs[layer][c_idx] = sign.item()
        return signs
        
        
        
#     def _compute_AGOP_concat(self, data, labels, llm, model, tokenizer, concat_layers,
#                              rep_token , hidden_state , hyperparams ):
        
#         n_components = hyperparams['n_components']
#         train_indices, val_indices = split_indices(len(data))
        
#         all_y = labels.float().cuda()
#         train_y = all_y[train_indices]
#         val_y = all_y[val_indices]
#         num_classes = all_y.shape[1]
        
#         direction_outputs = {'val' : [],'test' : []}
#         assert len(concat_layers)==2
                
        

#         hidden_states = direction_utils.get_hidden_states(data, llm, model, tokenizer, concat_layers, rep_token= rep_token, forward_batch_size = hyperparams['forward_batch_size'], hidden_state = hidden_state)
        
#         concatenated_states = torch.cat([hidden_states[concat_layers[0]],
#                                          hidden_states[concat_layers[1]]], dim=1).cuda().float()
                        
#         n_components = hyperparams['n_components']
#         assert n_components==1
#         directions = {}
      
#         train_X = concatenated_states[train_indices] 
#         val_X = concatenated_states[val_indices]

#         assert(len(train_X) == len(train_y))
#         assert(len(val_X) == len(val_y))

#         u,r,M = direction_utils.train_rfm_probe_on_concept(train_X, train_y, val_X, val_y)
#         u = u.reshape(1, -1)

#         signs = {}
#         if num_classes == 1: # only if binary do you compute signs   
#             sign = self._compute_signs({'layer_pair': concatenated_states}, all_y, {'layer_pair':u}, n_components)
#             u *= sign['layer_pair'][0]

#         return u, M

    
    
class PCAToolkit():
    def __init__(self):
        pass

    def _compute_directions(self, data, labels, use_soft_labels, 
                            llm, model, tokenizer, hidden_layers, 
                            rep_token, hyperparams):
        
                  
        print("Using soft labels: ", use_soft_labels)
        n_components = hyperparams['n_components']
        train_indices, val_indices = split_indices(len(data))
        num_classes = labels.shape[1]
        
       
        hidden_states, layer_to_soft_labels = direction_utils.get_hidden_states_and_attns(data, labels,llm, model,
                                                                                  tokenizer, hidden_layers,  rep_token)
                                                       
        
            
        assert hidden_states[1].shape[0] == layer_to_soft_labels[1].shape[0]
      
            
                
            
            
        train_indices, val_indices = split_indices(len(hidden_states[1]))
        print("shape of hidden states: ", hidden_states[1].shape)
        
        
        
        directions = {}
        rs = {}
        signs = {}

        for layer_to_eval in tqdm(hidden_layers):
            hidden_states_at_layer = hidden_states[layer_to_eval].cuda().float()
            all_y = layer_to_soft_labels[layer_to_eval].float().cuda() if use_soft_labels else labels.float().cuda()
            
            train_y = labels[train_indices] #must be binary for PCA training
            val_y = all_y[val_indices]
            
            
            train_X = hidden_states_at_layer[train_indices] 
            val_X = hidden_states_at_layer[val_indices]
                
            if layer_to_eval==2:
                print("train X shape:", train_X.shape, "train y shape:", train_y.shape, 
                      "val X shape:", val_X.shape, "val y shape:", val_y.shape)
                print("val_y: ", val_y)
            assert(len(train_X) == len(train_y))
            assert(len(val_X) == len(val_y))
       
            concept_features = direction_utils.fit_pca_model(train_X, train_y, 
                                                             val_X, val_y, n_components) # assumes the data are ordered in pos/neg pairs
            directions[layer_to_eval] = concept_features
            
            assert(concept_features.shape == (n_components, train_X.size(1)))
            
        signs = self._compute_signs(hidden_states, all_y, directions)
        for layer_to_eval in tqdm(hidden_layers):
            c_idx=0
            directions[layer_to_eval][c_idx] *= signs[layer_to_eval][c_idx]
        
        return directions, None, hidden_states, layer_to_soft_labels

    def _compute_signs(self, hidden_states, all_y, directions):
        
        signs = {}
        for layer in hidden_states.keys():
            xs = hidden_states[layer]
            signs[layer] = {}
            c_idx = 0
            direction = directions[layer][c_idx]
            hidden_state_projections = direction_utils.project_onto_direction(xs, direction).to(all_y.device)
            # print("hidden_state_projections", hidden_state_projections.shape, "all_y", all_y.shape)
            sign = 2*(direction_utils.pearson_corr(all_y.squeeze(1), hidden_state_projections) > 0) - 1
            signs[layer][c_idx] = sign.item()

        return signs
    
    
class LinearProbeToolkit():
    def __init__(self):
        pass

    def _compute_directions(self, data, labels, use_soft_labels, 
                            llm, model, tokenizer, hidden_layers, 
                            rep_token, hyperparams):
        
                  
        print("Using soft labels: ", use_soft_labels)
        n_components = hyperparams['n_components']
        train_indices, val_indices = split_indices(len(data))
        num_classes = labels.shape[1]
        
       
        hidden_states, layer_to_soft_labels = direction_utils.get_hidden_states_and_attns(data, labels,llm, model,
                                                                                  tokenizer, hidden_layers,  rep_token)
                                                       
        
            
        assert hidden_states[hidden_layers[0]].shape[0] == layer_to_soft_labels[hidden_layers[0]].shape[0]
      
            
                
        train_indices, val_indices = split_indices(len(hidden_states[hidden_layers[0]]))
        print("shape of hidden states: ", hidden_states[hidden_layers[0]].shape)
       
        
        
        directions = {}
        rs = {}
        signs = {}

        for layer_to_eval in tqdm(hidden_layers):
            hidden_states_at_layer = hidden_states[layer_to_eval].cuda().float()
            all_y = layer_to_soft_labels[layer_to_eval].float().cuda() if use_soft_labels else labels.float().cuda()
            
            train_y = all_y[train_indices]
            val_y = all_y[val_indices]
            
            
            train_X = hidden_states_at_layer[train_indices] 
            val_X = hidden_states_at_layer[val_indices]
                
            if layer_to_eval==2:
                print("train X shape:", train_X.shape, "train y shape:", train_y.shape, 
                      "val X shape:", val_X.shape, "val y shape:", val_y.shape)
                print("val_y: ", val_y)
            assert(len(train_X) == len(train_y))
            assert(len(val_X) == len(val_y))
            
            beta, bias = direction_utils.train_linear_probe_on_concept(train_X, train_y, val_X, val_y)
            
            assert(len(beta)==train_X.shape[1])
            if num_classes == 1: # assure beta is (num_classes, num_features)
                beta = beta.reshape(1,-1) 
            else:
                beta = beta.T
            beta /= beta.norm(dim=1, keepdim=True)
            directions[layer_to_eval] = beta
            
            ### Generate direction accuracy
            # solve for slope, intercept on training data
            # vec = beta.T
            # projected_train = train_X@vec
            # m, b = direction_utils.linear_solve(projected_train, train_y)                
            # detector_coefs[layer_to_eval] = [m, b]
            
            
        
        print("Computing signs")
        signs = {}
        if num_classes == 1: # only if binary do you compute signs
            signs = self._compute_signs(hidden_states, all_y, directions)
            for layer_to_eval in tqdm(hidden_layers):
                directions[layer_to_eval][0] *= signs[layer_to_eval][0] # only one direction, index 0
        
        
        return directions, None ,hidden_states, layer_to_soft_labels

    def _compute_signs(self, hidden_states, all_y, directions):
        
        signs = {}
        for layer in hidden_states.keys():
            xs = hidden_states[layer]
            signs[layer] = {}
            c_idx = 0
            direction = directions[layer][c_idx]
            hidden_state_projections = direction_utils.project_onto_direction(xs, direction).to(all_y.device)
            sign = 2*(direction_utils.pearson_corr(all_y.squeeze(1), hidden_state_projections) > 0) - 1
            signs[layer][c_idx] = sign.item()

        return signs
    
# class LinearProbeToolkit():
#     def __init__(self):
#         pass

#     def _compute_directions(self, data, labels, llm, model, tokenizer, hidden_layers, rep_token, hidden_state, layer_to_token, hyperparams):
        
                  
#         n_components = hyperparams['n_components']
#         train_indices, val_indices = split_indices(len(data))
        
#         all_y = labels.float().cuda()
#         train_y = all_y[train_indices]
#         val_y = all_y[val_indices]
#         num_classes = all_y.shape[1]
        
#         direction_outputs = {'val' : [],'test' : []}
        
#         hidden_states = direction_utils.get_hidden_states(data, llm, model, 
#                                                           tokenizer, hidden_layers, rep_token,
#                                                           16, verbose = False, hidden_state = hidden_state,
#                                                          layer_to_token=layer_to_token)
            
#         directions = {}
#         signs = {}

#         for layer_to_eval in tqdm(hidden_layers):
#             hidden_states_at_layer = hidden_states[layer_to_eval].cuda().float()
#             train_X = hidden_states_at_layer[train_indices] 
#             val_X = hidden_states_at_layer[val_indices]
                
#             print("linear: train X shape:", train_X.shape, "train y shape:", train_y.shape, 
#                   "val X shape:", val_X.shape, "val y shape:", val_y.shape)
#             assert(len(train_X) == len(train_y))
#             assert(len(val_X) == len(val_y))
            
            
#             beta, bias = direction_utils.train_linear_probe_on_concept(train_X, train_y, val_X, val_y)
            
#             assert(len(beta)==train_X.shape[1])
#             if num_classes == 1: # assure beta is (num_classes, num_features)
#                 beta = beta.reshape(1,-1) 
#             else:
#                 beta = beta.T
#             beta /= beta.norm(dim=1, keepdim=True)
#             directions[layer_to_eval] = beta
            
#             ### Generate direction accuracy
#             # solve for slope, intercept on training data
#             # vec = beta.T
#             # projected_train = train_X@vec
#             # m, b = direction_utils.linear_solve(projected_train, train_y)                
#             # detector_coefs[layer_to_eval] = [m, b]
            
            
        
#         print("Computing signs")
#         signs = {}
#         if num_classes == 1: # only if binary do you compute signs
#             signs = self._compute_signs(hidden_states, all_y, directions)
#             for layer_to_eval in tqdm(hidden_layers):
#                 directions[layer_to_eval][0] *= signs[layer_to_eval][0] # only one direction, index 0
        
        
        
#         return directions, None #, detector_coefs, None, None

#     def _compute_signs(self, hidden_states, all_y, directions):
        
#         signs = {}
#         for layer in hidden_states.keys():
#             xs = hidden_states[layer]
#             signs[layer] = {}
#             c_idx = 0
#             direction = directions[layer][c_idx]
#             hidden_state_projections = direction_utils.project_onto_direction(xs, direction).to(all_y.device)
#             sign = 2*(direction_utils.pearson_corr(all_y.squeeze(1), hidden_state_projections) > 0) - 1
#             signs[layer][c_idx] = sign.item()

#         return signs

    
class LogisticRegressionToolkit():
    def __init__(self):
        pass

    def _compute_directions(self, data, labels, use_soft_labels, 
                            llm, model, tokenizer, hidden_layers, 
                            rep_token, hyperparams):
        
                  
        print("Using soft labels: ", use_soft_labels)
        n_components = hyperparams['n_components']
        train_indices, val_indices = split_indices(len(data))
        num_classes = labels.shape[1]
        
       
        hidden_states, layer_to_soft_labels = direction_utils.get_hidden_states_and_attns(data, labels,llm, model,
                                                                                  tokenizer, hidden_layers,  rep_token)
                                                       
        
            
        assert hidden_states[1].shape[0] == layer_to_soft_labels[1].shape[0]
      
            
                
        train_indices, val_indices = split_indices(len(hidden_states[1]))
        print("shape of hidden states: ", hidden_states[1].shape)
        
        
        
        directions = {}
        rs = {}
        signs = {}

        for layer_to_eval in tqdm(hidden_layers):
            hidden_states_at_layer = hidden_states[layer_to_eval].cuda().float()
            all_y = layer_to_soft_labels[layer_to_eval].float().cuda() if use_soft_labels else labels.float().cuda()
            
            train_y = all_y[train_indices]
            val_y = all_y[val_indices]
            
            
            train_X = hidden_states_at_layer[train_indices] 
            val_X = hidden_states_at_layer[val_indices]
                
            if layer_to_eval==2:
                print("train X shape:", train_X.shape, "train y shape:", train_y.shape, 
                      "val X shape:", val_X.shape, "val y shape:", val_y.shape)
                print("val_y: ", val_y)
            assert(len(train_X) == len(train_y))
            assert(len(val_X) == len(val_y))
            
            # print("Training logistic regression")
            # Tune over Cs
            Cs = [1000, 10, 1, 1e-1]
            best_coef = None
            best_loss = float("inf")
            
            train_X_np = train_X.cpu().numpy()
            val_X_np = val_X.cpu().numpy()

            if num_classes == 1:
                train_y_flat = train_y.squeeze(1).cpu().numpy()
                val_y_flat = val_y.squeeze(1).cpu().numpy()
            else:
                train_y_flat = train_y.argmax(dim=1).cpu().numpy()
                val_y_flat = val_y.argmax(dim=1).cpu().numpy()

            # start = time.time()
            for C in Cs: 
                model = LogisticRegression(C=C, fit_intercept=False, 
                                           solver='liblinear', tol=1e-3)          
                model.fit(train_X_np, train_y_flat)
                val_probs = model.predict_proba(val_X_np)
                val_loss = log_loss(val_y_flat, val_probs)

                # print(f"Val loss: {val_loss}")
                if best_loss > val_loss: 
                    best_loss = val_loss
                    best_coef = model.coef_.copy()
            # end = time.time()
            # print("Logistic time: ", end - start)
            concept_features = torch.from_numpy(best_coef).to(train_X.dtype)

            if num_classes == 1:
                concept_features = concept_features.reshape(1,-1)

            assert(concept_features.shape == (num_classes, train_X.size(1)))
            concept_features /= concept_features.norm(dim=1, keepdim=True)

            directions[layer_to_eval] = concept_features
                                        
        print("Computing signs")
        signs = {}
        if num_classes == 1: # only if binary do you compute signs
            signs = self._compute_signs(hidden_states, all_y, directions)
            for layer_to_eval in tqdm(hidden_layers):
                directions[layer_to_eval][0] *= signs[layer_to_eval][0] # only one direction, index 0
            
        return directions, None, hidden_states, layer_to_soft_labels

    def _compute_signs(self, hidden_states, all_y, directions):
        
        signs = {}
        for layer in hidden_states.keys():
            xs = hidden_states[layer]
            signs[layer] = {}
            c_idx = 0
            direction = directions[layer][c_idx]
            hidden_state_projections = direction_utils.project_onto_direction(xs, direction).to(all_y.device)
            sign = 2*(direction_utils.pearson_corr(all_y.squeeze(1), hidden_state_projections) > 0) - 1
            signs[layer][c_idx] = sign.item()

        return signs

    
class MeanDifferenceToolkit():
    def __init__(self):
        pass

    def _compute_directions(self, data, labels, use_soft_labels, 
                            llm, model, tokenizer, hidden_layers, 
                            rep_token, hyperparams):
        
                  
        print("Using soft labels: ", use_soft_labels)
        n_components = hyperparams['n_components']
        train_indices, val_indices = split_indices(len(data))
        num_classes = labels.shape[1]
        
       
        hidden_states, layer_to_soft_labels = direction_utils.get_hidden_states_and_attns(data, labels,llm, model,
                                                                                  tokenizer, hidden_layers,  rep_token)
                                                       
            
        assert hidden_states[hidden_layers[0]].shape[0] == layer_to_soft_labels[hidden_layers[0]].shape[0]
      
            
                
        train_indices, val_indices = split_indices(len(hidden_states[hidden_layers[0]]))
        print("shape of hidden states: ", hidden_states[hidden_layers[0]].shape)
        
        
        
        directions = {}
        rs = {}
        signs = {}

        for layer_to_eval in tqdm(hidden_layers):
            hidden_states_at_layer = hidden_states[layer_to_eval].cuda().float()
            all_y = layer_to_soft_labels[layer_to_eval].float().cuda() if use_soft_labels else labels.float().cuda()
            
            train_y = labels[train_indices]
            # val_y = all_y[val_indices]
            assert ((train_y == 0) | (train_y == 1)).all(), "Training labels must be 0 or 1"
            
            
            train_X = hidden_states_at_layer[train_indices] 
            # val_X = hidden_states_at_layer[val_indices]
                
            if layer_to_eval==2:
                print("train X shape:", train_X.shape, "train y shape:", train_y.shape, 
                      "val X shape:", val_X.shape, "val y shape:", val_y.shape)
                print("val_y: ", val_y)
            assert(len(train_X) == len(train_y))
            # assert(len(val_X) == len(val_y))
            
                        
            pos_indices = torch.isclose(train_y, torch.ones_like(train_y)).squeeze(1)
            neg_indices = torch.isclose(train_y, torch.zeros_like(train_y)).squeeze(1)
            
            pos_mean = train_X[pos_indices].mean(dim=0)
            neg_mean = train_X[neg_indices].mean(dim=0)
            
            concept_features = pos_mean - neg_mean
            concept_features /= concept_features.norm()
            
            directions[layer_to_eval] = concept_features.reshape(1,-1)
            
                      
                
        
        return directions, None,  hidden_states, layer_to_soft_labels
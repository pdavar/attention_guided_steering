import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import read_file, get_csv_filename
import itertools

hidden_state = 'block'
rep_token_idx_to_token = {-4: 'start_header_id', -3:'assistant', -2:'end_header_id',-1:'newline'}
model_type = 'llama_3.1_70b'
print("===== Model: ", model_type)

dataset_to_lower = {'fears':True, 
                        'personalities':True, 
                        'moods':True, 
                        'places':False, 
                        'personas':False}


methods = ['rfm', 'linear', 'logistic', 'mean_difference','pca']
concept_classes = ['fears', 'places','personas','personalities','moods']
rep_tokens = [-4,-3,-2,-1,"max_attn_per_layer","max_attn_per_layer_softlabels", 'max_per_layer_avgstmnt']
records = []  # collect results

rows = pd.MultiIndex.from_product([methods, concept_classes ], names=["method", "concept_class"])
cols = pd.Index(rep_tokens, name="rep_token")
df = pd.DataFrame(index=rows, columns=cols, dtype=float)

# Fill values

for  method, concept_class in itertools.product(methods,concept_classes):
    for rep_token in rep_tokens:
        fname = f"data/concepts/{concept_class}.txt" 
        n_concepts = len(read_file(fname, lower=dataset_to_lower[concept_class]))
        scores = []
        all_5_versions = True
        for version in (1, 2, 3, 4, 5):
            if rep_token=='max_per_layer_avgstmnt':
                csv_path = f'data/archived results/{method}_{concept_class}_tokenidxmax_per_layer_avgstmnt_block_gpt4o_outputs_500_concepts_{model_type}_{version}.csv'
            elif rep_token=="max_attn_per_layer_softlabels": 
                csv_path = get_csv_filename(method, concept_class, 'max_attn_per_layer', model_type, version, True)
            else:
                csv_path = get_csv_filename(method, concept_class, rep_token, model_type, version, False)
            try:
                s = pd.read_csv(csv_path)["best_score"].astype(float).values

                assert len(s)== n_concepts, f'csv file is incomplete for {method} {concept_class} {rep_token} version {version}'
                scores.extend(s)
            except:
                scores.extend([float('nan')] *  n_concepts )
                all_5_versions = False
        if all_5_versions: df.loc[(method, concept_class), rep_token] = np.mean(scores)

# Set hierarchical index
df.to_csv(f"data/csvs/{model_type}_all_scores.csv",float_format="%.3g")


import re
import csv
import pandas as pd
from openai import OpenAI
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import sys
import utils
import os
from args import get_args

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)




dataset_to_lower = {'fears':True, 
                        'personalities':True, 
                        'moods':True, 
                        'places':False, 
                        'personas':False}
                     

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)


def main(rep_token, concept_class, method, model_type, use_soft_labels, version): 

    fname = f"data/concepts/{concept_class}.txt"        
    n_concepts = len(utils.read_file(fname, lower=dataset_to_lower[concept_class]))
    
    
    outpath = utils.get_csv_filename(method, concept_class, rep_token, model_type, version, use_soft_labels)
    file_path = utils.get_steered_output_filename(method, concept_class, rep_token, model_type, version, use_soft_labels)

    if not os.path.exists(file_path):
        print("file doesn't exist! \n", file_path)
        raise ValueError
    p = utils.safe_load_pickle(file_path)
    if p is None or (p is not None and len(p) != n_concepts):
        print("file doesn't have all outputs! \n", file_path)
        print('length: ', len(pickle.load(open(file_path, 'rb'))))
        raise ValueError


    if os.path.exists(outpath) and pd.read_csv(outpath).shape[0] == n_concepts:
        print("File already exists")
        return
    print("Path does not exist... initiating API call")
    starting_idx = 0
    outputs = []


    results = pickle.load(open(file_path, 'rb'))
    assert n_concepts == len(list(results.keys())), f"the saved outputs doesn't have all concepts: {file_path}"

    for results_idx, concept in enumerate(list(results.keys())[starting_idx:]):
        print(concept)
        responses = results[concept]#[concat_layers_tuple]
        assert len(responses)>5, f"concept {concept} version {version} mehtod {method} doesn't have all outputs"
        print(f"============================= CONCEPT:{concept} ===========================")
        best_score = 0
        best_coef = 0

        for response in responses:
            parsed_response = ""
            parsed_response = utils.parse_personality_responses(response, model_type)
            print("parsed response: ", parsed_response)
            if parsed_response == "":
                parsed_response = "None"

            prompt_template = utils.load_prompt(concept_class, version)

            prompt = prompt_template.format(personality=concept, parsed_response=parsed_response)            

            output = client.chat.completions.create(
                    messages=[
                        { "role": "user",
                            "content": prompt}],
                    temperature=0.,
                    max_tokens=20,
                    model='gpt-4o-2024-11-20' 
                )
            content = output.choices[0].message.content
            score = 0

            # print(prompt)
            if "Score: " in content:
                try:                        
                    score = int(content.split("Score: ")[1][0])
                except (IndexError, ValueError) as e:
                    score = 0
            if score>best_score:
                best_score = score
                best_coef = response[0]
            print(f"\nscore = {score} for llama response for coef {response[0]}: {parsed_response}")
            print("gpt output: ", content)
        outputs.append((concept, best_score, best_coef))



    df = pd.DataFrame(outputs, columns=["mood", "best_score", "best_coef"])

    # Save to CSV
    df.to_csv(outpath, index=False)


if __name__ == "__main__":
    
    rep_token, model_type, dataset_label, method, version, label= get_args()
    use_soft_labels = label=='soft'
    main(rep_token, dataset_label, method, model_type, use_soft_labels, version)
    
    

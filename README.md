## Attention-guided steering 

Python scripts `0_visualize_attn.py`–`4_visualize_scores.py` implement the full workflow for extracting attention-to-prefix, extracting concept vectors, generating steered outputs, evaluating them with GPT-4o, and summarizing scores.

### 1) Environment
- Python ≥3.10 and an NVIDIA GPU with CUDA (models load in 4-bit when available).
```
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

pip install -r requirements.txt
```
- Set your OpenAI key for evaluations (needed for script `3_evaluate_steered_outputs.py`): `export OPENAI_API_KEY="<your-key-here>"`.
- Set Hugging Face auth if the chosen model requires it (`huggingface-cli login`).
- Required data already lives under `data/`. Attention-to-prefixes, directions (steering vectors), and steered outputs are written beside it (e.g., `data/attention_to_prompt`, `data/directions`, `data/cached_outputs`).
- To steer towrads a custom concept that is not already listed under the 5 concept classes in `data/concepts`, use the flag `-c custom` and insert your entire prefix in the `data/concepts/custom.txt` file.
- To test the pipeline on only the first 5 concepts in each concept class, set `run_first_five = True` at the top of the scripts. 

### 2) Shared CLI flags
`args.py` defines common flags (defaults in parentheses):
- `--rep_token/-t` (`max_attn_per_layer`): token position or strategy for representation.
- `--model_name/-m` (`llama_3.1_8b`): see `utils.select_llm` for allowed IDs.
- `--concept_type/-c` (`fears`): one of `fears|personalities|moods|places|personas|jailbreaking|custom`.
- `--control_method/-cm` (`rfm`): steering method.
- `--version/-v` (`1`): test prompt version.
- `--label/-l` (`soft`): `soft` or `hard` labels.

### 3) Pipeline steps
- **0_visualize_attn.py** – Generates a numpy array containing the attention-to-prefix for all final tokens shared by all prompts.

```
python 0_visualize_attn.py -t max_attn_per_layer -m llama_3.1_8b -c fears -cm rfm -v 1 -l soft
```
  - Output: `data/attention_to_prompt/attentions_meanhead_<model>_<concept>_paired_statements.npy`.

- **1_get_directions.py** – train and save concept vectors (directions) per concept.

```
python 1_get_directions.py -t max_attn_per_layer -m llama_3.1_8b -c fears -cm rfm -v 1 -l soft
```
  - Output: direction vectors saved under `/data/directions/`.

- **2_steer.py** – generate original + steered completions for each concept and coefficient.

```
python 2_steer.py -t max_attn_per_layer -m llama_3.1_8b -c fears -cm rfm -v 1 -l soft
```
  - Output: pickled generations at `data/cached_outputs/<method>_<concept>_tokenidx<rep>_block[_softlabels]_steered_500_concepts_<model>_<version>.pkl`.

- **3_evaluate_steered_outputs.py** – evaluated steered outputs with GPT-4o and pick best coef per concept.
  - Requires `OPENAI_API_KEY` in the environment.
```
OPENAI_API_KEY=... python 3_evaluate_steered_outputs.py -t max_attn_per_layer -m llama_3.1_8b -c fears -cm rfm -v 1 -l soft
```
  - Output: CSV written to `data/csvs/<method>_<concept>_tokenidx<rep>_block[_softlabels]_gpt4o_outputs_500_concepts_<model>_<version>.csv`.

- **4_visualize_scores.py** – aggregate per-method/per-rep-token scores into a summary table.
```
python 4_visualize_scores.py
```
  - Output: `data/csvs/<model>_all_scores.csv`.

### 4) Typical end-to-end run
```
# 0) compute attention stats (needed when using max_attn_per_layer)
python 0_visualize_attn.py -m llama_3.1_8b -c fears

# 1) learn concept vectors
python 1_get_directions.py -m llama_3.1_8b -c fears

# 2) generate steered outputs
python 2_steer.py -m llama_3.1_8b -c fears

# 3) score with GPT-4o
OPENAI_API_KEY=... python 3_evaluate_steered_outputs.py -m llama_3.1_8b -c fears

# 4) summarize scores
python 4_visualize_scores.py
```

### 5) Notes
- GPU memory: 70B models need substantial VRAM; if constrained, use `llama_3.1_8b` or adjust model choice.
- Data: concept lists live in `data/concepts/`; general statements in `data/general_statements/`; prompt templates in `data/evaluation_prompts/`.
- Determinism: seeds are set inside scripts, but generation can still vary across hardware/drivers.

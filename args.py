import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Run steering with specified token, model, and concept type.")

    parser.add_argument(
        "--rep_token", "-t",
        default="max_attn_per_layer",
        help="Token used for extracting hidden states(int or string)."
    )

    parser.add_argument(
        "--model_name", "-m",
        default="llama_3.1_8b",
        help="Model name to use (default: llama_3.1_8b)."
    )

    parser.add_argument(
        "--concept_type", "-c",
        default="fears",
        help="Concept type (default: fears)."
    )
    
    parser.add_argument(
        "--control_method", "-cm",
        default="rfm",
        help="Control method (default: rfm)."
    )
    
    parser.add_argument(
        "--version", "-v",
        default="1",
        help="version of test prompts to use"
    )
    parser.add_argument(
        "--label", "-l",
        default="soft",
        help="using hard or soft labels"
    )

    args = parser.parse_args()

    # Convert rep_token to int if possible
    try:
        rep_token = int(args.rep_token)
    except ValueError:
        rep_token = args.rep_token

    return rep_token, args.model_name, args.concept_type, args.control_method, args.version, args.label

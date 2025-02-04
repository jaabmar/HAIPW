import argparse
import pandas as pd
from importlib import import_module
from HAIPW.utils import setup_logging, log
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["gpt4o", "claude_haiku", "deepseek", "llama", "llama_small"])
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--study_name", type=str, required=True, default="faheyS78")
    args = parser.parse_args()
    log_file = f"../logs/data_{args.model_name}_{args.study_name}.log"
    setup_logging(log_file)

    study_name = args.study_name
    df = pd.read_csv(f"{study_name}/df_processed.csv")

    if args.n_samples is not None:
        df = df.sample(n=args.n_samples, random_state=42)

    module_name = ("generate_outcomes_propietary" if args.model_name in {"gpt4o", "claude_haiku", "deepseek"}
                   else "generate_outcomes_opensource")

    generate_synthetic_data = import_module(f"{study_name}.{module_name}").generate_synthetic_data
    df_llm, mse_y1, mse_y0 = generate_synthetic_data(df, args.model_name)

    log("Synthetic Data Generation Complete")
    df_llm.to_csv(f"{study_name}/df_{args.model_name}.csv", index=False)

    # Compute mse means
    mse_y0_means = {k: np.mean(v) if v else None for k, v in mse_y0.items()}
    mse_y1_means = {k: np.mean(v) if v else None for k, v in mse_y1.items()}

    # Log results
    log(f"mse_y0 (mean values): {json.dumps(mse_y0_means, indent=4)}")
    log(f"mse_y1 (mean values): {json.dumps(mse_y1_means, indent=4)}")


if __name__ == "__main__":
    main()

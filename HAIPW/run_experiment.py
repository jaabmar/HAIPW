import argparse
import ast
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
from HAIPW.utils import setup_logging, log, load_dataframe, compute_coverage
from HAIPW.estimators import AIPWEstimator, HAIPWEstimator, PPIEstimator, DifferenceInMeansEstimator


def run_experiment(df_augmented, models, n_features, n_rct, n_folds, alpha_ridge, gt, seed):
    df_augmented = df_augmented.groupby('T').sample(n=n_rct // 2, random_state=seed)
    Y_rct = df_augmented["Y"].to_numpy()
    T_rct = df_augmented["T"].to_numpy()
    X_rct = df_augmented.drop(columns=["T", "Y", "Unnamed: 0"]
                              + [f"Y1_{model}" for model in models]
                              + [f"Y0_{model}" for model in models]).to_numpy()

    # Initialize estimators
    aipw_estimator = AIPWEstimator(alpha_ridge, n_folds, n_features)
    haipw_estimator = HAIPWEstimator(alpha_ridge, n_folds, n_features)
    ppi_estimator = PPIEstimator()
    dim_estimator = DifferenceInMeansEstimator()

    # Compute estimates and variances
    aipw_est, aipw_var = aipw_estimator.estimate(X_rct, Y_rct, T_rct)

    model_predictions_y1 = [df_augmented[f"Y1_{model}"].to_numpy() for model in models]
    model_predictions_y0 = [df_augmented[f"Y0_{model}"].to_numpy() for model in models]
    haipw_est, haipw_var = haipw_estimator.estimate(X_rct, Y_rct, T_rct, model_predictions_y1, model_predictions_y0)

    ppi_est, ppi_var = ppi_estimator.estimate(X_rct, Y_rct, T_rct, df_augmented[f"Y0_{models[0]}"])
    dm_est, dm_var = dim_estimator.estimate(X_rct, Y_rct, T_rct)

    # Compute coverage
    coverage_aipw = compute_coverage(aipw_est, aipw_var, gt, n_rct)
    coverage_haipw = compute_coverage(haipw_est, haipw_var, gt, n_rct)
    coverage_ppi = compute_coverage(ppi_est, ppi_var, gt, n_rct)
    coverage_dm = compute_coverage(dm_est, dm_var, gt, n_rct)

    return {
        "aipw_est": aipw_est, "aipw_var": aipw_var, "coverage_aipw": coverage_aipw,
        "haipw_est": haipw_est, "haipw_var": haipw_var, "coverage_haipw": coverage_haipw,
        "ppi_est": ppi_est, "ppi_var": ppi_var, "coverage_ppi": coverage_ppi,
        "dm_est": dm_est, "dm_var": dm_var, "coverage_dm": coverage_dm
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiments")
    parser.add_argument("--n_rct", type=int, default=30, help="Number of samples in RCT")
    parser.add_argument("--n_features", type=int, default=5)
    parser.add_argument("--n_folds", type=int, default=20)
    parser.add_argument("--alpha_ridge", type=float, default=0.1)
    parser.add_argument("--study", type=str, required=True)
    parser.add_argument("--model", nargs="+", type=str, required=True, help="Models: e.g., 'gpt4o llama claude_haiku'")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--n_prompts", type=int, default=1)

    args = parser.parse_args()
    log_file = f"../logs/experiment_{args.study}.log"
    setup_logging(log_file)

    log(f"Running experiment for study: {args.study}")
    log(f"Models: {', '.join(args.model)}")
    log(f"RCT Samples: {args.n_rct}, Features: {args.n_features}, Folds: {args.n_folds}")
    log(f"Alpha Ridge: {args.alpha_ridge}, Seeds: {args.n_seeds}, Prompts: {args.n_prompts}")

    df = load_dataframe(f"{args.study}/df_processed.csv")
    Y = df["Y"].to_numpy()
    T = df["T"].to_numpy()
    ground_truth = Y[T == 1].mean() - Y[T == 0].mean()
    df_full = df.copy()

    for model in args.model:
        df_model = load_dataframe(f"{args.study}/df_{model}.csv")[["Y0hat_responses", "Y1hat_responses"]]
        Y0_model, Y1_model = [], []

        for _, row in df_model.iterrows():
            Y0_list = ast.literal_eval(row["Y0hat_responses"])
            Y1_list = ast.literal_eval(row["Y1hat_responses"])
            sampled_Y0 = Y0_list[:args.n_prompts]
            sampled_Y1 = Y1_list[:args.n_prompts]
            Y0_model.append(np.mean(sampled_Y0))
            Y1_model.append(np.mean(sampled_Y1))

        df_model[f"Y0_{model}"] = Y0_model
        df_model[f"Y1_{model}"] = Y1_model
        df_model.drop(columns=["Y0hat_responses", "Y1hat_responses"], inplace=True)
        df_full = pd.concat([df_model, df_full], axis=1)

    n_jobs = 50
    log(f'running n_jobs: {n_jobs}')

    def run_single_exp(seed):
        return run_experiment(df_full, args.model, args.n_features, args.n_rct, args.n_folds,
                              args.alpha_ridge, ground_truth, seed)

    with Pool(n_jobs) as pool:
        results = list(tqdm(pool.imap(run_single_exp, range(args.n_seeds)), total=args.n_seeds))

    # Aggregate results
    metrics = {key: np.mean([res[key] for res in results]) for key in results[0].keys()}

    # Log results
    for key, value in metrics.items():
        log(f"{key}: {value}")

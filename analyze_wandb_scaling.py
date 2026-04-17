import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

ENTITY = "wl_meet_rad"
PROJECT = "wl_rad"

OUTPUT_DIR = "analysis_results_acc"


###########################################
# Fetch runs from WandB
###########################################

def fetch_runs():

    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    rows = []

    for run in runs:

        if run.state != "finished":
            continue

        summary = run.summary
        config = run.config

        if "gen_error_upper_bound" not in summary:
            continue

        history = run.history(keys=["gen_err_acc"], pandas=True)

        if len(history) == 0:
            continue

        gen_error = history["gen_err_acc"].dropna().iloc[-1]

        rows.append({

            "dataset": config.get("dataset", "NCI1"),
            "layers": config.get("num_layers"),
            "seed": config.get("seed"),
            "m": config.get("m"),

            "R_s": summary["R_s_upper_bound"],
            "gen_bound": summary["gen_error_upper_bound"],
            "gen_error": abs(gen_error)

        })

    df = pd.DataFrame(rows)

    return df


###########################################
# Aggregate across seeds
###########################################

def aggregate(df):

    grouped = df.groupby(["dataset","layers","m"]).agg({

        "gen_error":["mean","std"],
        "gen_bound":["mean","std"],
        "R_s":"mean"

    }).reset_index()

    grouped.columns = [
        "dataset","layers","m",
        "gen_mean","gen_std",
        "bound_mean","bound_std",
        "R_s"
    ]

    return grouped


###########################################
# Scaling tests
###########################################

def scaling_tests(df):

    slope_m,_,r_m,_,_ = linregress(
        np.log(df["m"]),
        np.log(df["gen_mean"])
    )

    slope_bound,_,r_b,_,_ = linregress(
        np.log(df["bound_mean"]),
        np.log(df["gen_mean"])
    )

    df["Rs_over_m"] = df["R_s"]/df["m"]

    slope_rs,_,r_rs,_,_ = linregress(
        np.log(df["Rs_over_m"]),
        np.log(df["gen_mean"])
    )

    return {
        "m_exponent": slope_m,
        "m_r2": r_m**2,
        "bound_exponent": slope_bound,
        "bound_r2": r_b**2,
        "rs_exponent": slope_rs,
        "rs_r2": r_rs**2
    }


###########################################
# Plotting
###########################################

def plot_scaling_vs_m(df, dataset, layers, outdir):

    plt.figure()

    plt.errorbar(
        df["m"],
        df["gen_mean"],
        yerr=df["gen_std"],
        fmt="o"
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Sample size m")
    plt.ylabel("Generalization Error")

    plt.title(f"{dataset} | layers={layers}\nGen Error vs m")

    plt.tight_layout()

    plt.savefig(os.path.join(outdir,"scaling_vs_m.png"))

    plt.close()


def plot_bound_vs_empirical(df, dataset, layers, outdir):

    plt.figure()

    plt.scatter(df["bound_mean"], df["gen_mean"])

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Theoretical Bound")
    plt.ylabel("Empirical Gen Error")

    plt.title(f"{dataset} | layers={layers}\nBound vs Empirical Error")

    plt.tight_layout()

    plt.savefig(os.path.join(outdir,"bound_vs_empirical.png"))

    plt.close()


def plot_rs_scaling(df, dataset, layers, outdir):

    plt.figure()

    df["Rs_over_m"] = df["R_s"]/df["m"]

    plt.scatter(df["Rs_over_m"], df["gen_mean"])

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("R_s / m")
    plt.ylabel("Generalization Error")

    plt.title(f"{dataset} | layers={layers}\nStructure Scaling")

    plt.tight_layout()

    plt.savefig(os.path.join(outdir,"rs_scaling.png"))

    plt.close()


###########################################
# Save scaling report
###########################################

def save_report(results, dataset, layers, outdir):

    with open(os.path.join(outdir,"scaling_report.txt"),"w") as f:

        f.write(f"Dataset: {dataset}\n")
        f.write(f"Layers: {layers}\n\n")

        f.write("Scaling vs m\n")
        f.write(f"Exponent: {results['m_exponent']}\n")
        f.write(f"R2: {results['m_r2']}\n\n")

        f.write("Scaling vs Bound\n")
        f.write(f"Exponent: {results['bound_exponent']}\n")
        f.write(f"R2: {results['bound_r2']}\n\n")

        f.write("Scaling vs Rs/m\n")
        f.write(f"Exponent: {results['rs_exponent']}\n")
        f.write(f"R2: {results['rs_r2']}\n")


###########################################
# Main analysis
###########################################

def main():

    os.makedirs(OUTPUT_DIR,exist_ok=True)

    df = fetch_runs()

    print("Runs loaded:",len(df))

    grouped = aggregate(df)

    for dataset in grouped["dataset"].unique():

        df_dataset = grouped[grouped["dataset"]==dataset]

        for layers in sorted(df_dataset["layers"].unique()):

            df_case = df_dataset[df_dataset["layers"]==layers]

            if len(df_case) < 3:
                continue

            outdir = os.path.join(
                OUTPUT_DIR,
                dataset,
                f"layers_{layers}"
            )

            os.makedirs(outdir,exist_ok=True)

            results = scaling_tests(df_case)

            plot_scaling_vs_m(df_case,dataset,layers,outdir)

            plot_bound_vs_empirical(df_case,dataset,layers,outdir)

            plot_rs_scaling(df_case,dataset,layers,outdir)

            save_report(results,dataset,layers,outdir)

            print(f"Finished {dataset} layers={layers}")


if __name__ == "__main__":
    main()

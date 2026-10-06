from pathlib import Path
from fire import Fire
import os
import json
import datetime
from dataclasses import dataclass
from typing import List
from tabulate import tabulate, SEPARATING_LINE
from statistics import mean

from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_point,
    facet_grid,
    facet_wrap,
    scale_y_continuous,
    geom_hline,
    position_dodge,
    geom_errorbar,
    scale_y_discrete,
    theme,
    element_text,
    ylab,
    xlab,
    scale_color_discrete,
)
from pandas import Categorical, DataFrame
from plotnine.scales.limits import ylim
from plotnine.scales.scale_xy import scale_x_discrete

@dataclass
class ReliabilityResult:
    section: str
    ece: float
    mce: float
    train_samples: int
    samples: int
    model_type: str
    file_path: str
    gstd: float = None
    iv_model: bool = False


def find_reliability_results(root, gstd_filter, train_samples_filter):
    results = []
    for subdir, _, files in os.walk(root):
        for file in files:
            if file.startswith("reliability_") and file.endswith(".json"):
                full_path = os.path.join(subdir, file)
                with open(full_path, "r") as f:
                    data = json.load(f)
                    for section in ["normal", "scaled_down", "uncertainty_ui1"]:
                        if section in data:
                            section_data = data[section]
                            ece = section_data.get("ece", None)
                            mce = section_data.get("mce", None)
                            # Parse samples from filename, e.g., ...results_s2... means samples=2
                            import re
                            samples = None
                            samples_match = re.search(r"results_s(\d+)", full_path)
                            if samples_match:
                                samples = int(samples_match.group(1))
                            full_model_type_str = full_path.split("/")[-2]
                            model_type, train_samples_str = full_model_type_str.split("_") if "_" in full_model_type_str else (full_model_type_str, None)
                            train_samples = int(train_samples_str[1:]) if train_samples_str else None

                            # Parse gstd from the path if present as /gstd_5/
                            gstd = None
                            match = re.search(r"/gstd_([\d.]+)/", full_path)
                            if match:
                                gstd = float(match.group(1))
                            if "ua" in model_type:
                                continue
                            if gstd is not None and gstd_filter is not None and gstd not in gstd_filter:
                                continue
                            if train_samples is not None and train_samples_filter is not None and train_samples not in train_samples_filter:
                                continue
                            iv_model = "iv_" in full_path
                            results.append(
                                ReliabilityResult(
                                    section=section,
                                    ece=ece,
                                    mce=mce,
                                    samples=samples,
                                    train_samples=train_samples,
                                    model_type=model_type,
                                    file_path=full_path,
                                    gstd=gstd,
                                    iv_model=iv_model
                                )
                            )
    return results


def show_reliability_table(results):
    results = sorted(results, key=lambda r: r.ece)
    headers = ["Model Type", "Section", "ECE", "MCE", "Samples", "gstd", "iv_model", "Plot"]
    table = [headers, SEPARATING_LINE]
    def get_plot_path(r):
        # Replace results root with plots folder and .json with .png
        plot_path = r.file_path.replace("results-reliability-with-uncertainty", "plots-reliability-with-uncertainty").replace(".json", ".png")
        return plot_path
    for r in results:
        line = [r.model_type, r.section, f"{r.ece:.4f}", f"{r.mce:.4f}", r.samples, r.gstd, r.iv_model, get_plot_path(r)]
        table.append(line)
    print(tabulate(table))

    model_name_map = {
        "gat": "GAT",
        "gcn": "GCN",
        "vgat": "VGAT",
        "vgcn": "VGCN",
        "dropoutgcn": "Dropout-GCN",
        "dropoutgat": "Dropout-GAT",
    }

    df = DataFrame([
        {
            "Model Type": model_name_map[r.model_type],
            "Section": r.section,
            "ECE": r.ece,
            "MCE": r.mce,
            "Test Samples": r.samples,
            "gstd": r.gstd,
            "iv_model": r.iv_model,
            "Plot": get_plot_path(r),
        }
        for r in results
    ])
    return df

def plot_reliability_scores(df, output_file_name):
    # Map section names to capitalized, no-underscore versions for plotting (optional, but keep for facet labels)
    section_map = {
        "normal": "Normal",
        "scaled_down": "Scaled-down",
        "uncertainty_ui1": "Uncertainty-scaled"
    }
    df["Section"] = df["Section"].map(section_map)
    df["Section"] = Categorical(df["Section"], ordered=True, categories=["Normal", "Uncertainty-scaled", "Scaled-down"])
    df["Variance Scale"] = df["gstd"].astype(str).map(lambda x: x if x != "nan" else "-")
    df = df.sort_values(by="iv_model")
    df["Initialized Models"] = df["iv_model"].astype(str).map(lambda x: "IVGAT/IVGCN" if x == "True" else "Other Models")

    model_order = reversed(["GCN", "Dropout-GCN", "VGCN", "IVGCN", "GAT", "Dropout-GAT", "VGAT", "IVGAT"])
    df["Model Type"] = Categorical(df["Model Type"], ordered=True, categories=model_order)

    plot = (
        ggplot(df)
        + aes(x="ECE", y="Model Type", color="Test Samples", shape="Initialized Models")
        + geom_point(size=2, alpha=1)
        + facet_wrap(["Section"], ncol=1)
        + theme(figure_size=(4, 4.5), strip_text_x=element_text(size=8))
    )
    plot.save(str(output_file_name), dpi=300)

    plot_mce = (
        ggplot(df)
        + aes(x="MCE", y="Model Type", color="Test Samples", shape="Initialized Models")
        + geom_point(size=2, alpha=1)
        + facet_wrap(["Section"], ncol=1)
        + theme(figure_size=(4, 4.5), strip_text_x=element_text(size=8))
    )
    plot_mce.save(str(output_file_name).replace(".png", "_mce.png"), dpi=300)


# Used to create reliability figures (4 in the paper)
def main(
    root="./results-reliability-with-uncertainty",
    plots_folder="plots-reliability-with-uncertainty",
    plot_name="reliability_scores",
    gstd_filter=None,
    train_samples_filter=None,
):
    results = find_reliability_results(root, gstd_filter, train_samples_filter)
    df = show_reliability_table(results)
    Path(plots_folder).mkdir(exist_ok=True)
    plot_reliability_scores(df, Path(plots_folder) / f"{plot_name}.png")


if __name__ == "__main__":
    Fire()

from pathlib import Path
from fire import Fire
import os
import re
from dataclasses import dataclass
from typing import List
import json
from tabulate import tabulate, SEPARATING_LINE
import datetime
import simple_colors as colors
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
from plotnine.data import economics
from pandas import Categorical, DataFrame
from plotnine.scales.limits import ylim
from plotnine.scales.scale_xy import scale_x_discrete

HORIZON_FLAGS = ["Lead-lag", "Simultaneous", "h_mean"]
FREQUENCY_FLAGS = ["D", "W", "f_mean"]
DIRECTION_FLAGS = ["Buy", "Sell", "d_mean"]
MODEL_TYPE_FLAGS = [
    "baselines",
    "vnn_gat",
    "vnn_gcn",
    "dropout_gcn",
    "dropout_gat",
    "vnn_gat_ivb",
    "vnn_gcn_ivb",
    "vnn_gat_ivo",
    "vnn_gcn_ivo",
    "uavgat",
    "uaeavgat",
    "uafmcivgat",
    "original",
]


def is_mean_flag(flag):
    return flag in ["h_mean", "f_mean", "d_mean"]


def transform_model_type_for_plotting(model_type):
    if model_type in ["vnn_gat"]:
        return "vgat"
    if model_type in ["vnn_gcn"]:
        return "vgcn"

    if model_type in ["dropout_gcn"]:
        return "dropoutgcn"

    if model_type in ["vnn_gat_ivb", "vnn_gat_ivo"]:
        return "ivgat"
    if model_type in ["vnn_gcn_ivb", "vnn_gcn_ivo"]:
        return "ivgcn"

    if model_type in ["dropout_gat"]:
        return "dropoutgat"

    return model_type


def transform_subset_name_for_plotting(name):
    if "_mean" in name:
        return "Mean"

    return name


@dataclass
class SingleResult:
    samples: int
    batch: int
    f1: float
    f1_std: float
    auc: float
    non_own_f1: float
    non_own_auc: float
    insiders_non_own_f1: float
    insiders_non_own_auc: float


@dataclass
class Experiment:
    network_type: str
    samples: int
    batch: int
    flags: List[str]
    results: List[SingleResult]
    age_days: int
    path: str
    __best_result: float = None

    def best_result(self):

        if self.__best_result is None:
            self.__best_result = max([r for r in self.results], key=lambda r: r.f1)

        return self.__best_result

    def __str__(self):
        result = f"Experiment(network_type={self.network_type}, samples={self.samples}, batch={self.batch}, age_days={self.age_days}, flags={self.flags}\n"

        for r in self.results:
            result += f"    {r}\n"

        result += ")"
        return result


def file_age_in_days(path):
    return (
        datetime.datetime.today()
        - datetime.datetime.fromtimestamp(os.path.getmtime(path))
    ).days


def group_experiments(experiments: List[Experiment]):

    result = {}

    for horizon in HORIZON_FLAGS:
        for frequency in FREQUENCY_FLAGS:
            for direction in DIRECTION_FLAGS:
                for model_type in MODEL_TYPE_FLAGS:
                    if horizon not in result:
                        result[horizon] = {}
                    if frequency not in result[horizon]:
                        result[horizon][frequency] = {}
                    if direction not in result[horizon][frequency]:
                        result[horizon][frequency][direction] = {}
                    if model_type not in result[horizon][frequency][direction]:
                        result[horizon][frequency][direction][model_type] = []

    for experiment in experiments:
        horizon = None
        frequency = None
        direction = None
        model_type = None
        iv = None

        for flag in experiment.flags:
            if flag in HORIZON_FLAGS:
                horizon = flag
            elif flag in FREQUENCY_FLAGS:
                frequency = flag
            elif flag in DIRECTION_FLAGS:
                direction = flag
            elif flag in MODEL_TYPE_FLAGS:
                model_type = flag
            elif isinstance(flag, tuple) and flag[0] == "iv_base":
                iv = flag[1]

        if iv:
            model_type += "_iv" + iv[0]
            if model_type not in MODEL_TYPE_FLAGS:
                MODEL_TYPE_FLAGS.append(model_type)

        if (
            (horizon is not None)
            and (frequency is not None)
            and (direction is not None)
            and (model_type is not None)
        ):
            if horizon not in result:
                result[horizon] = {}
            if frequency not in result[horizon]:
                result[horizon][frequency] = {}
            if direction not in result[horizon][frequency]:
                result[horizon][frequency][direction] = {}
            if model_type not in result[horizon][frequency][direction]:
                result[horizon][frequency][direction][model_type] = []
            result[horizon][frequency][direction][model_type].append(experiment)

    return result


def create_mean_results(experiments: List[Experiment]):

    result = {}

    def is_dataset_flag(flag):
        return (
            (flag in HORIZON_FLAGS)
            or (flag in FREQUENCY_FLAGS)
            or (flag in DIRECTION_FLAGS)
        )

    def mean_of_results(results: List[SingleResult]):
        return SingleResult(
            results[0].samples,
            results[0].batch,
            mean([r.f1 for r in results]),
            mean([r.f1_std for r in results]),
            mean([r.auc for r in results]),
            mean([r.non_own_f1 for r in results]),
            mean([r.non_own_auc for r in results]),
            mean([r.insiders_non_own_f1 for r in results]),
            mean([r.insiders_non_own_auc for r in results]),
        )

    for experiment in experiments:
        sorted_flags = sorted(
            [str(f) for f in experiment.flags if not is_dataset_flag(f)]
        )
        combined_flags = (
            "_".join(sorted_flags)
            + f"{experiment.network_type}_{experiment.samples}_{experiment.batch}"
        )

        if combined_flags not in result:
            result[combined_flags] = []

        result[combined_flags].append(experiment)

    mean_set = []

    for experiment_set in result.values():
        mean_experiment = Experiment(
            experiment_set[0].network_type,
            experiment_set[0].samples,
            experiment_set[0].batch,
            [*experiment_set[0].flags, "h_mean", "f_mean", "d_mean"],
            [
                mean_of_results([r.results[i] for r in experiment_set])
                for i in range(len(experiment_set[0].results))
            ],
            (
                min(*[a.age_days for a in experiment_set])
                if len(experiment_set) > 1
                else experiment_set[0].age_days
            ),
            path=experiment_set[0].path,
        )

        mean_set.append(mean_experiment)

    return mean_set


def show_inclusion_table(
    experiments: List[Experiment],
    experiments_per_group,
    exclude_model_types=[],
    allow_model_types=None,
    show_empty=True,
    frame_extra_flags=[],
):

    experiments = experiments + create_mean_results(experiments)

    extra_flags = set()
    value_flags = set()

    for i, experiment in enumerate(experiments):
        
        if i % 5000 == 0:
            print(".", end="")

        for flag in experiment.flags:

            if isinstance(flag, tuple):
                flag = flag[0]
                value_flags.add(flag)

            if not (
                (flag in HORIZON_FLAGS)
                or (flag in FREQUENCY_FLAGS)
                or (flag in DIRECTION_FLAGS)
                or (flag in MODEL_TYPE_FLAGS)
            ):
                extra_flags.add(flag)

    extra_flags = list(extra_flags)

    groups = group_experiments(experiments)

    headers = [
        "horizon",
        "freq",
        "dir",
        "model type",
        "network",
        "f1",
        "f1 std",
        "auc",
        "no f1",
        "no auc",
        "ino f1",
        "ino auc",
        "s",
        "ts",
        *extra_flags,
    ]
    table = []
    raw_table = []

    base_data_frame_entries = [
        "horizon",
        "frequency",
        "direction",
        "Model Type",
        "Network",
        "F1",
        "f1 std",
        "auc",
        "s",
        "ts",
    ]

    data_frame = {}

    for f in base_data_frame_entries:
        data_frame[f] = []

    for f in frame_extra_flags:
        data_frame[f[1]] = []

    def colored_line(color, line):

        if color is None:
            return line

        return [color(a) for a in line]

    def experiment_color(experiment: Experiment):

        color = None

        if experiment.age_days <= 0:
            color = colors.magenta
        elif experiment.age_days <= 3:
            color = colors.green
        elif experiment.age_days <= 7:
            color = colors.yellow

        return color

    table.append(colored_line(colors.cyan, headers))
    raw_table.append(headers)
    table.append(SEPARATING_LINE)
    raw_table.append(SEPARATING_LINE)

    last_table_len = 2

    for horizon in HORIZON_FLAGS:
        for frequency in FREQUENCY_FLAGS:
            for direction in DIRECTION_FLAGS:
                if (
                    (is_mean_flag(horizon))
                    or (is_mean_flag(frequency))
                    or (is_mean_flag(direction))
                ):
                    if not (
                        (is_mean_flag(horizon))
                        and (is_mean_flag(frequency))
                        and (is_mean_flag(direction))
                    ):
                        continue

                print(".", end="")

                for model_type in MODEL_TYPE_FLAGS:
                    if model_type in exclude_model_types:
                        continue
                    if (
                        allow_model_types is not None
                        and model_type not in allow_model_types
                    ):
                        continue

                    experiments = groups[horizon][frequency][direction][model_type]

                    def has_filtered_flags(exp: Experiment):
                        for f in exp.flags:
                            if isinstance(f, tuple):
                                if f[0] == "iv_base" and (
                                    f[1] in exclude_model_types
                                    or (
                                        allow_model_types is not None
                                        and f[1] in allow_model_types
                                    )
                                ):
                                    return True
                        return False

                    experiments = [
                        exp
                        for exp in experiments
                        if (not has_filtered_flags(exp))
                        and not (exp.network_type in exclude_model_types)
                        # and (allow_model_types is None or exp.network_type in allow_model_types)
                    ]

                    experiments = sorted(
                        experiments,
                        key=lambda experiment: -experiment.best_result().f1,
                    )

                    groups[horizon][frequency][direction][model_type] = experiments

    for horizon in HORIZON_FLAGS:
        for frequency in FREQUENCY_FLAGS:
            for direction in DIRECTION_FLAGS:
                if (
                    (is_mean_flag(horizon))
                    or (is_mean_flag(frequency))
                    or (is_mean_flag(direction))
                ):
                    if not (
                        (is_mean_flag(horizon))
                        and (is_mean_flag(frequency))
                        and (is_mean_flag(direction))
                    ):
                        continue

                for model_type in MODEL_TYPE_FLAGS:
                    if model_type in exclude_model_types:
                        continue
                    if (
                        allow_model_types is not None
                        and model_type not in allow_model_types
                    ):
                        continue

                    experiments_exist = False
                    experiments = groups[horizon][frequency][direction][model_type]

                    for i, experiment in enumerate(experiments):

                        if i >= experiments_per_group:
                            line = [
                                horizon,
                                frequency,
                                direction,
                                model_type,
                                "...",
                                "...",
                                "...",
                                "...",
                                "...",
                                "...",
                                "...",
                                "...",
                                "...",
                                # "...",
                                "...",
                                *["..." for _ in extra_flags],
                            ]
                            table.append(line)
                            raw_table.append(line)
                            break

                        experiments_exist = True

                        best_result = experiment.best_result()

                        def flag_to_text(f):
                            if f in value_flags:
                                for ef in experiment.flags:
                                    if isinstance(ef, tuple) and ef[0] == f:
                                        return ef[1]
                                return ""
                            else:
                                return "+" if f in experiment.flags else ""

                        is_best_in_subset = False

                        if i == 0:
                            is_best_in_subset = True
                            for compare_model_type in MODEL_TYPE_FLAGS:
                                if (model_type != compare_model_type) and (
                                    compare_model_type not in exclude_model_types
                                ):

                                    if (
                                        len(
                                            groups[horizon][frequency][direction][
                                                compare_model_type
                                            ]
                                        )
                                        <= 0
                                    ):
                                        continue

                                    compare_experiment = groups[horizon][frequency][
                                        direction
                                    ][compare_model_type][0]
                                    if (
                                        compare_experiment.best_result().f1
                                        > best_result.f1
                                    ):
                                        is_best_in_subset = False

                        line = [
                            horizon,
                            frequency,
                            direction,
                            model_type,
                            experiment.network_type,
                            f"{best_result.f1:.3f}"
                            + ("#" if i == 0 else "")
                            + ("##" if is_best_in_subset else ""),
                            (
                                f"{best_result.f1_std:.3f}"
                                if best_result.f1_std != 0
                                else ""
                            ),
                            f"{best_result.auc:.2f}",
                            f"{best_result.non_own_f1:.2f}",
                            f"{best_result.non_own_auc:.2f}",
                            f"{best_result.insiders_non_own_f1:.2f}",
                            f"{best_result.insiders_non_own_auc:.2f}",
                            ("" if experiment.samples is None else experiment.samples),
                            # ("" if experiment.batch is None else experiment.batch),
                            ("" if best_result.samples == -1 else best_result.samples),
                            *[flag_to_text(f) for f in extra_flags],
                        ]
                        table.append(colored_line(experiment_color(experiment), line))
                        raw_table.append(line)

                        data_frame["horizon"].append(
                            transform_subset_name_for_plotting(horizon)
                        )
                        data_frame["frequency"].append(
                            transform_subset_name_for_plotting(frequency)
                        )
                        data_frame["direction"].append(
                            transform_subset_name_for_plotting(direction)
                        )
                        data_frame["Model Type"].append(
                            transform_model_type_for_plotting(model_type)
                        )
                        data_frame["Network"].append(experiment.network_type)
                        data_frame["F1"].append(best_result.f1)
                        data_frame["f1 std"].append(best_result.f1_std)
                        data_frame["auc"].append(best_result.auc)
                        data_frame["s"].append(
                            0 if experiment.samples is None else experiment.samples
                        )
                        data_frame["ts"].append(
                            0 if best_result.samples == -1 else best_result.samples
                        )
                        for f in frame_extra_flags:
                            if f[1] not in base_data_frame_entries:
                                data_frame[f[1]].append(flag_to_text(f[1]))

                    if (not experiments_exist) and show_empty:

                        line = [
                            horizon,
                            frequency,
                            direction,
                            model_type,
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            # "",
                            "",
                            "",
                            *["" for _ in extra_flags],
                        ]
                        table.append(line)
                        raw_table.append(line)

                table.append(SEPARATING_LINE)
                raw_table.append(SEPARATING_LINE)

            if len(table) - last_table_len > 20:
                table.append(colored_line(colors.cyan, headers))
                raw_table.append(headers)
                table.append(SEPARATING_LINE)
                raw_table.append(SEPARATING_LINE)

                last_table_len = len(table)

    tab = tabulate(table)
    raw_tab = tabulate(raw_table)
    print(tab)
    with open("inclusion_table.txt", "w") as f:
        print(raw_tab, file=f)

    data_frame = DataFrame(data_frame)

    return data_frame


def plot_single_frame(frame, output_file_name, model_type_order, plots_extra_flags=[]):

    extra_aes = {"color": "Network"}
    for k, v in plots_extra_flags:
        extra_aes[k] = v

    frame.horizon = Categorical(
        frame.horizon, ordered=True, categories=["Lead-lag", "Simultaneous", "Mean"]
    )
    
    frame = frame.sort_values("F1")

    plot = (
        ggplot(frame)
        + aes(x="F1", y="Model Type")
        + facet_wrap(["horizon", "frequency", "direction"], ncol=2)
        + geom_point(
            aes(**extra_aes),
            size=1.5,
            # position=position_dodge(width=0.8),
            # stroke=0.2,
        )
        + scale_y_discrete(limits=model_type_order)
    )

    plot = plot + theme(figure_size=(4, 8), strip_text_x=element_text(size=5))

    plot.save(str(output_file_name), dpi=600)


def find_experiments(
    results_root, root, allowed_subfolders=None, forbidden_subfolders=None
):

    subdirs = os.walk(root)

    all_result_files = []

    for subdir, _, files in subdirs:
        for file in files:
            if "result.json" in file:
                if allowed_subfolders is None or any(
                    [a in subdir for a in allowed_subfolders]
                ):
                    if forbidden_subfolders is None or not any(
                        [f in subdir for f in forbidden_subfolders]
                    ):
                        all_result_files.append((subdir, file))

    print(f"Found {len(all_result_files)} results")

    all_experiments = []

    for subdir, file in all_result_files:
        full_file_name = os.path.join(subdir, file)
        # print(full_file_name)

        groups = subdir.replace(results_root + "/", "").split("/")
        network_group = groups.pop()
        network_type, *params = re.findall(r"-?[\d\.]+|[a-zA-Z-]+", network_group)
        samples = None
        batch = None
        flags = []
        iv_type = None

        i = 0

        for g in groups:
            if g in MODEL_TYPE_FLAGS:
                params.append(g)
            else:
                params += re.findall(r"-?[\d\.]+|[a-zA-Z-]+", g)

        while i < len(params):
            if params[i] == "s":
                samples = int(params[i + 1])
                i += 1
            elif params[i] in ["seed", "seeds"]:
                flags.append(("seed", params[i + 1]))
                i += 1
            elif params[i] == "b":
                batch = int(params[i + 1])
                i += 1
            elif params[i] == "activation":
                flags.append(("activ", params[i + 1]))
                i += 1
            elif params[i] == "gstd-mode":
                flags.append(("gstd-m", params[i + 1]))
                i += 1
            elif params[i] == "dropout-probability":
                flags.append(("drop-p", params[i + 1]))
                i += 1
            elif params[i] == "dropout-type":
                flags.append(("drop-t", params[i + 1]))
                i += 1
            elif params[i] == "gstd":
                flags.append(("gstd", params[i + 1]))
                i += 1
            elif params[i] == "train":
                flags.append(("uatrain", params[i + 1]))
                i += 1
            elif params[i] == "afl":
                flags.append(("afl", params[i + 1]))
                i += 1
            elif params[i] in ["iv"]:
                flags.append(("iv_base", params[i + 1]))
                i += 1
            elif params[i] in ["xufb", "xnfb"]:
                iv_type = params[i] + params[i + 1] + params[i + 2] + params[i + 3]
                flags.append(("iv_type", iv_type))
                i += 3
            elif params[i] in ["xu", "xn"]:
                iv_type = (
                    params[i]
                    + params[i + 1]
                    + params[i + 2]
                    + params[i + 3]
                    + params[i + 4]
                    + params[i + 5]
                )
                flags.append(("iv_type", iv_type))
                i += 5
            elif params[i] in ["f"]:
                iv_type = params[i] + params[i + 1] + params[i + 2] + params[i + 3]
                flags.append(("iv_type", iv_type))
                i += 3
            elif params[i] in ["usual"]:
                iv_type = params[i]
                flags.append(("iv_type", iv_type))
                i += 1
            else:
                flags.append(params[i])
            i += 1

        experiment_results = []

        with open(full_file_name, "r") as f:

            data = json.load(f)

            if not isinstance(data, list):
                data = [data]

            for d in data:
                single_result = SingleResult(
                    samples=d["test_samples"] if "test_samples" in d else -1,
                    batch=d["batch"] if "batch" in d else -1,
                    f1=d["f1"],
                    f1_std=d["f1_std"] if "f1_std" in d else 0,
                    auc=d["auc"],
                    non_own_f1=d["non_own_f1"],
                    non_own_auc=d["non_own_auc"],
                    insiders_non_own_f1=d["insiders_non_own_f1"],
                    insiders_non_own_auc=d["insiders_non_own_auc"],
                )

                experiment_results.append(single_result)

        experiment = Experiment(
            network_type=network_type,
            samples=samples,
            batch=batch,
            flags=flags,
            results=experiment_results,
            age_days=file_age_in_days(full_file_name),
            path=full_file_name,
        )

        all_experiments.append(experiment)

    return all_experiments


def main(
    root="./results",
    plots_folder="plots",
    exclude_model_types=[],
    allow_model_types=None,
    experiments_per_group=10,
    plots_extra_flags=[],
    plot_name="insider-results",
    model_type_order=[
        "original",
        "ivgat",
        "vgat",
        "dropoutgat",
        "ivgcn",
        "vgcn",
        "dropoutgcn",
        "baselines",
    ],
):

    if not isinstance(exclude_model_types, list):
        exclude_model_types = [exclude_model_types]

    if not isinstance(allow_model_types, list):
        allow_model_types = [allow_model_types] if allow_model_types is not None else None

    all_experiments = find_experiments(root, root)

    data_frame = show_inclusion_table(
        all_experiments,
        exclude_model_types=exclude_model_types,
        allow_model_types=allow_model_types,
        experiments_per_group=experiments_per_group,
        frame_extra_flags=plots_extra_flags,
    )

    for t in exclude_model_types:
        if t in model_type_order:
            model_type_order.remove(t)

    plot_single_frame(
        data_frame,
        Path(plots_folder) / f"{plot_name}.png",
        model_type_order,
        plots_extra_flags=plots_extra_flags,
    )


def uncertainty_aware(
    ua_models_root="./results/vnn_gat/iv_baselines_xnfb0x2/activation_mean/gstd-mode_multiply",
    vnn_models_root="./results/vnn_gat/iv_baselines_xnfb0x2/activation_mean/gstd-mode_multiply",
    results_root="./results",
    ua_model_types=["train_variational", "train_uncertainty_aware"],
    samples=None,
    plots_folder="plots",
    exlcude_model_types=[],
    experiments_per_group=100,
):

    if not isinstance(exlcude_model_types, list):
        exlcude_model_types = [exlcude_model_types]

    ua_experiments = find_experiments(
        results_root, ua_models_root, allowed_subfolders=ua_model_types
    )
    vnn_experiments = find_experiments(
        results_root, vnn_models_root, forbidden_subfolders=ua_model_types
    )

    if samples is not None:
        ua_experiments = [e for e in ua_experiments if e.samples == samples]
        vnn_experiments = [e for e in vnn_experiments if e.samples == samples]

    all_experiments = ua_experiments + vnn_experiments

    data_frame = show_inclusion_table(
        all_experiments,
        exclude_model_types=exlcude_model_types,
        experiments_per_group=experiments_per_group,
    )

    # model_type_order = ["original", "ivnn", "vnn", "baselines"]
    model_type_order = [
        "original",
        "ivgat",
        "vgat",
        "ivgcn",
        "vgcn",
        "dropoutgcn",
        "dropoutgat",
        "baselines",
    ]

    for t in exlcude_model_types:
        if t in model_type_order:
            model_type_order.remove(t)

    plot_single_frame(
        data_frame,
        Path(plots_folder) / "uncertainty_aware-results.png",
        model_type_order,
    )


if __name__ == "__main__":
    Fire()

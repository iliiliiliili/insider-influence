from fire import Fire
import os
import re
from dataclasses import dataclass
from typing import List
import json
from tabulate import tabulate, SEPARATING_LINE
import datetime
import simple_colors as colors

HORIZON_FLAGS = ["Lead-lag", "Simultaneous"]
FREQUENCY_FLAGS = ["D", "W"]
DIRECTION_FLAGS = ["Buy", "Sell"]
MODEL_TYPE_FLAGS = ["baselines", "vnn", "original"]


@dataclass
class SingleResult:
    samples: int
    batch: int
    f1: float
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

    def best_result(self):
        return max([r for r in self.results], key=lambda r: r.f1)

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

        for flag in experiment.flags:
            if flag in HORIZON_FLAGS:
                horizon = flag
            elif flag in FREQUENCY_FLAGS:
                frequency = flag
            elif flag in DIRECTION_FLAGS:
                direction = flag
            elif flag in MODEL_TYPE_FLAGS:
                model_type = flag

        if (
            (horizon is not None)
            and (frequency is not None)
            and (direction is not None)
            and (model_type is not None)
        ):
            result[horizon][frequency][direction][model_type].append(experiment)

    return result


def show_inclusion_table(experiments: List[Experiment], show_empty=True):

    extra_flags = set()
    value_flags = set()

    for experiment in experiments:
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
        "auc",
        "no f1",
        "no auc",
        "ino f1",
        "ino auc",
        "samples",
        "batch",
        "test samples",
        *extra_flags,
    ]
    table = []
    raw_table = []

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
                for model_type in MODEL_TYPE_FLAGS:

                    experiments_exist = False

                    experiments = groups[horizon][frequency][direction][model_type]
                    experiments = sorted(
                        experiments,
                        key=lambda experiment: -experiment.best_result().f1,
                    )

                    for i, experiment in enumerate(experiments):

                        experiments_exist = True

                        best_result = experiment.best_result()

                        def flag_to_text(f):
                            if f in value_flags:
                                for ef in experiment.flags:
                                    if isinstance(ef, tuple) and ef[0] == f:
                                        return ef[1]
                            else:
                                return "+" if f in experiment.flags else ""

                        is_best_in_subset = False
                        
                        if i == 0:
                            is_best_in_subset = True
                            for compare_model_type in MODEL_TYPE_FLAGS:
                                if model_type != compare_model_type:
                                    for compare_experiment in groups[horizon][frequency][direction][compare_model_type]:
                                        if compare_experiment.best_result().f1 > best_result.f1:
                                            is_best_in_subset = False

                        line = [
                            horizon,
                            frequency,
                            direction,
                            model_type,
                            experiment.network_type,
                            f"{best_result.f1:.2f}" + ("#" if i == 0 else "") + ("##" if is_best_in_subset else ""),
                            f"{best_result.auc:.2f}",
                            f"{best_result.non_own_f1:.2f}",
                            f"{best_result.non_own_auc:.2f}",
                            f"{best_result.insiders_non_own_f1:.2f}",
                            f"{best_result.insiders_non_own_auc:.2f}",
                            ("" if experiment.samples is None else experiment.samples),
                            ("" if experiment.batch is None else experiment.batch),
                            ("" if best_result.samples == -1 else best_result.samples),
                            *[flag_to_text(f) for f in extra_flags],
                        ]
                        table.append(colored_line(experiment_color(experiment), line))
                        raw_table.append(line)

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


def main(root="./results", show_inclusion=True):
    subdirs = os.walk(root)

    all_result_files = []

    for subdir, _, files in subdirs:
        for file in files:
            if "result.json" in file:
                all_result_files.append((subdir, file))

    print(f"Found {len(all_result_files)} results")

    all_experiments = []

    for subdir, file in all_result_files:
        full_file_name = os.path.join(subdir, file)
        print(full_file_name)

        groups = subdir.replace(root + "/", "").split("/")
        network_group = groups.pop()
        network_type, *params = re.findall(
            r"[a-zA-Z-]+|\d+", network_group
        )
        samples = None
        batch = None
        flags = []
        iv_type = None

        i = 0

        while i < len(params):
            if params[i] == "s":
                samples = int(params[i + 1])
                i += 1
            if params[i] == "seed":
                flags.append(("seed", params[i + 1]))
                i += 1
            if params[i] == "seeds":
                flags.append(("seed", params[i + 1], params[i + 2]))
                i += 2
            elif params[i] == "b":
                batch = int(params[i + 1])
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

        flags += groups

        experiment_results = []

        with open(full_file_name, "r") as f:

            data = json.load(f)

            if not isinstance(data, list):
                data = [data]
                
            for d in data:
                single_result = SingleResult(
                    samples=d["samples"] if "samples" in d else -1,
                    batch=d["batch"] if "batch" in d else -1,
                    f1=d["f1"],
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
        )

        all_experiments.append(experiment)

    for exp in all_experiments:
        print(exp)

    if show_inclusion:
        show_inclusion_table(all_experiments)


if __name__ == "__main__":
    Fire(main)

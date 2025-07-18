from fire import Fire
from main import main
from multiprocessing import Pool

from networks.variational import VariationalBase
import random


def vnn_base_name(vnn_name):
    if "gat" in vnn_name:
        return "gat"
    if "gcn" in vnn_name:
        return "gcn"

    return vnn_name


def run_experiments(experiments, devices, processes_per_device, debug=False):
    print(f"Starting {len(experiments)} experiments")

    experiments_per_device = [0 for _ in devices]

    experiments_to_assign = len(experiments)

    for i in range(len(experiments_per_device)):
        experiments_per_device[i] = min(experiments_to_assign, processes_per_device)
        experiments_to_assign -= experiments_per_device[i]

    while experiments_to_assign > 0:
        for i in range(len(experiments_per_device)):
            experiments_per_device[i] += 1
            experiments_to_assign -= 1

            if experiments_to_assign <= 0:
                break

    pools = [Pool(processes_per_device) for _ in devices]

    for exp_count, device, pool in zip(experiments_per_device, devices, pools):

        for i in range(exp_count):
            experiment_args, experiment_kwargs = experiments.pop(0)

            if debug:
                print("DEBUG")
                print("DEBUG")
                print("DEBUG")
                main(
                    *experiment_args,
                    device=device,
                    **experiment_kwargs,
                    override_horizon="Lead-lag",
                    override_frequency="D",
                    override_direction="Buy",
                )
            else:
                # pool.apply(
                pool.apply_async(
                    main,
                    args=(*experiment_args,),
                    kwds={"device": device, **experiment_kwargs},
                    error_callback=lambda e, args=experiment_args, kwargs=experiment_kwargs: open(
                        "./errors.log", "a"
                    ).write(
                        f"Error: {str(e)}\nArgs: {args}\nKwargs: {kwargs}\n"
                    ),
                )

    for pool in pools:
        pool.close()
        pool.join()

    print("Done")


def vnn(networks=["vgcn", "vgat"], devices=8, processes_per_device=3, debug=False):

    if not isinstance(devices, list):
        devices = [f"cuda:{d}" for d in range(devices)]

    experiments = []

    for net in networks:
        for samples in range(2, 10):
            experiments.append(
                (
                    ["train", f"n", net, f"vnn_{vnn_base_name(net)}"],
                    {"train_samples": samples},
                )
            )

    run_experiments(experiments, devices, processes_per_device, debug)


def vnn_all(networks=["vgcn", "vgat"], devices=8, processes_per_device=3, debug=False):

    if not isinstance(devices, list):
        devices = [f"cuda:{d}" for d in range(devices)]

    experiments = []

    for net in networks:
        for samples in range(2, 10):
            for activation_mode in VariationalBase.ALL_ACTIVATION_MODES:
                for use_batch_norm in [False, True]:
                    for global_std_mode in VariationalBase.ALL_GLOBAL_STD_MODES:

                        if global_std_mode == "none":
                            gstds = [0]
                        else:
                            gstds = [0, 0.1, 0.2, 0.5, 1, 2, 5]

                        for gstd in gstds:

                            name = (
                                f"activation_{activation_mode}{'/bn' if use_batch_norm else ''}/gstd-mode_{global_std_mode}"
                                + ("" if global_std_mode == "none" else f"/gstd_{gstd}")
                            )

                            experiments.append(
                                (
                                    ["train", name, net, f"vnn_{vnn_base_name(net)}"],
                                    {
                                        "train_samples": samples,
                                        "batch_norm_mode": activation_mode,
                                        "activation_mode": activation_mode,
                                        "use_batch_norm": use_batch_norm,
                                        "global_std_mode": global_std_mode,
                                        "GLOBAL_STD": gstd,
                                    },
                                )
                            )

    run_experiments(experiments, devices, processes_per_device, debug)


def vnn_init(networks=["vgcn", "vgat"], devices=8, processes_per_device=3, debug=False):

    if not isinstance(devices, list):
        devices = [f"cuda:{d}" for d in range(devices)]

    experiments = []

    for net in networks:
        for samples in range(2, 10):
            for activation_mode in VariationalBase.ALL_ACTIVATION_MODES:
                for use_batch_norm in [True, False]:
                    for global_std_mode in VariationalBase.ALL_GLOBAL_STD_MODES:
                        for init_from in ["baselines", "original"]:
                            for init_vnn_name, init_vnn_weights in [
                                ("usual", "usual"),
                                ("f0x3", "fill:stds:0.001:0.001"),
                                ("f0x4", "fill:stds:0.0001:0.0001"),
                                ("xu0b0x2", "xavier_uniform0b:stds:0.01:0.001"),
                                ("xufb0x2", "xavier_uniform_fb:stds:0.01:0.001"),
                                ("xnfb0x2", "xavier_normal_fb:stds:0.01:0.001"),
                                ("xn0b0x2", "xavier_uniform_0b:stds:0.01:0.001"),
                            ]:

                                if global_std_mode == "none":
                                    gstds = [0]
                                else:
                                    gstds = [0, 0.01, 0.05, 0.1, 0.2]

                                for gstd in gstds:

                                    name = (
                                        f"activation_{activation_mode}{'/bn' if use_batch_norm else ''}/gstd-mode_{global_std_mode}"
                                        + (
                                            ""
                                            if global_std_mode == "none"
                                            else f"/gstd_{gstd}"
                                        )
                                    )

                                    experiments.append(
                                        (
                                            [
                                                "train",
                                                name,
                                                net,
                                                f"vnn_{vnn_base_name(net)}/iv_{init_from}_{init_vnn_name}",
                                            ],
                                            {
                                                "train_samples": samples,
                                                "batch_norm_mode": activation_mode,
                                                "activation_mode": activation_mode,
                                                "use_batch_norm": use_batch_norm,
                                                "global_std_mode": global_std_mode,
                                                "GLOBAL_STD": gstd,
                                                "init_vnn_from": f"models/{init_from}/{vnn_base_name(net)}",
                                                "INIT_WEIGHTS": init_vnn_weights,
                                                "init_vnn_from_original": init_from
                                                == "original",
                                            },
                                        )
                                    )

    run_experiments(experiments, devices, processes_per_device, debug)


def base_experiment_all(
    dataset_suffix="",
    classical_networks=["gcn", "gat"],
    vnn_networks=["vgcn", "vgat"],
    devices=8,
    processes_per_device=3,
    debug=False,
):

    if not isinstance(devices, list):
        devices = [f"cuda:{d}" for d in range(devices)]

    experiments = []

    for net in classical_networks:
        experiments.append(
            (
                ["train", ".", net, f"baselines"],
                {
                    "dataset_folder": f"data{dataset_suffix}",
                    "results_folder": f"results{dataset_suffix}",
                    "models_folder": f"models{dataset_suffix}",
                },
            )
        )

    for net in vnn_networks:
        for samples in [2, 7]:
            for activation_mode in ["mean"]:
                for use_batch_norm in [False]:
                    for global_std_mode in ["multiply"]:

                        if global_std_mode == "none":
                            gstds = [0]
                        else:
                            gstds = [0, 0.01, 0.05, 0.1, 0.2]

                        for gstd in gstds:

                            name = (
                                f"activation_{activation_mode}{'/bn' if use_batch_norm else ''}/gstd-mode_{global_std_mode}"
                                + ("" if global_std_mode == "none" else f"/gstd_{gstd}")
                            )

                            experiments.append(
                                (
                                    ["train", name, net, f"vnn_{vnn_base_name(net)}"],
                                    {
                                        "train_samples": samples,
                                        "batch_norm_mode": activation_mode,
                                        "activation_mode": activation_mode,
                                        "use_batch_norm": use_batch_norm,
                                        "global_std_mode": global_std_mode,
                                        "GLOBAL_STD": gstd,
                                        "dataset_folder": f"data{dataset_suffix}",
                                        "results_folder": f"results{dataset_suffix}",
                                        "models_folder": f"models{dataset_suffix}",
                                    },
                                )
                            )

    for net in vnn_networks:
        for samples in [2, 7]:
            for activation_mode in ["mean"]:
                for use_batch_norm in [False]:
                    for global_std_mode in ["multiply"]:
                        for init_from in ["baselines", "original"]:
                            for init_vnn_name, init_vnn_weights in [
                                ("usual", "usual"),
                                ("f0x3", "fill:stds:0.001:0.001"),
                                ("f0x4", "fill:stds:0.0001:0.0001"),
                                ("xu0b0x2", "xavier_uniform0b:stds:0.01:0.001"),
                                ("xufb0x2", "xavier_uniform_fb:stds:0.01:0.001"),
                                ("xnfb0x2", "xavier_normal_fb:stds:0.01:0.001"),
                                ("xn0b0x2", "xavier_uniform_0b:stds:0.01:0.001"),
                            ]:

                                if global_std_mode == "none":
                                    gstds = [0]
                                else:
                                    gstds = [0, 0.01, 0.05, 0.1, 0.2]

                                for gstd in gstds:

                                    name = (
                                        f"activation_{activation_mode}{'/bn' if use_batch_norm else ''}/gstd-mode_{global_std_mode}"
                                        + (
                                            ""
                                            if global_std_mode == "none"
                                            else f"/gstd_{gstd}"
                                        )
                                    )

                                    experiments.append(
                                        (
                                            [
                                                "train",
                                                name,
                                                net,
                                                f"vnn_{vnn_base_name(net)}/iv_{init_from}_{init_vnn_name}",
                                            ],
                                            {
                                                "train_samples": samples,
                                                "batch_norm_mode": activation_mode,
                                                "activation_mode": activation_mode,
                                                "use_batch_norm": use_batch_norm,
                                                "global_std_mode": global_std_mode,
                                                "GLOBAL_STD": gstd,
                                                "init_vnn_from": f"models/{init_from}/{vnn_base_name(net)}",
                                                "INIT_WEIGHTS": init_vnn_weights,
                                                "init_vnn_from_original": init_from
                                                == "original",
                                                "dataset_folder": f"data{dataset_suffix}",
                                                "results_folder": f"results{dataset_suffix}",
                                                "models_folder": f"models{dataset_suffix}",
                                            },
                                        )
                                    )

    run_experiments(experiments, devices, processes_per_device, debug)


def uncertainty_aware(
    networks=["uavgat"], devices=8, processes_per_device=2, debug=False
):

    if not isinstance(devices, list):
        devices = [f"cuda:{d}" for d in range(devices)]

    experiments = []

    for net in networks:
        for samples in range(2, 5):
            for activation_mode in ["mean", "end"]:
                for use_batch_norm in [False]:
                    for global_std_mode in VariationalBase.ALL_GLOBAL_STD_MODES:
                        for init_from in ["baselines"]:
                            for init_vnn_name, init_vnn_weights in [
                                ("usual", "usual"),
                                ("f0x3", "fill:stds:0.001:0.001"),
                                ("f0x4", "fill:stds:0.0001:0.0001"),
                                ("xu0b0x2", "xavier_uniform_0b:stds:0.01:0.001"),
                                ("xufb0x2", "xavier_uniform_fb:stds:0.01:0.001"),
                                ("xnfb0x2", "xavier_normal_fb:stds:0.01:0.001"),
                                ("xn0b0x2", "xavier_uniform_0b:stds:0.01:0.001"),
                            ]:
                                for training_method in [
                                    "variational",
                                    "uncertainty_aware",
                                ]:

                                    if global_std_mode == "none":
                                        gstds = [0]
                                    else:
                                        gstds = [0, 0.01, 0.05, 0.1, 0.2]

                                    for gstd in gstds:

                                        name = (
                                            f"activation_{activation_mode}{'/bn' if use_batch_norm else ''}/gstd-mode_{global_std_mode}/train_{training_method}"
                                            + (
                                                ""
                                                if global_std_mode == "none"
                                                else f"/gstd_{gstd}"
                                            )
                                        )

                                        experiments.append(
                                            (
                                                [
                                                    "train",
                                                    name,
                                                    net,
                                                    f"vnn_{vnn_base_name(net)}/iv_{init_from}_{init_vnn_name}",
                                                ],
                                                {
                                                    "train_samples": samples,
                                                    "batch_norm_mode": activation_mode,
                                                    "activation_mode": activation_mode,
                                                    "use_batch_norm": use_batch_norm,
                                                    "global_std_mode": global_std_mode,
                                                    "GLOBAL_STD": gstd,
                                                    "init_vnn_from": f"models/{init_from}/{vnn_base_name(net)}",
                                                    "INIT_WEIGHTS": init_vnn_weights,
                                                    "init_vnn_from_original": init_from
                                                    == "original",
                                                    "training_method": training_method,
                                                },
                                            )
                                        )
    random.shuffle(experiments)
    run_experiments(experiments, devices, processes_per_device, debug)


def train_uncertainty_aware_from_best_vgat(
    networks=["uaeavgat"], devices=8, processes_per_device=1, debug=False
):

    if not isinstance(devices, list):
        devices = [f"cuda:{d}" for d in range(devices)]

    experiments = []

    for net in networks:
        for samples in [7, 2]:
            for activation_mode in ["mean"]:
                for use_batch_norm in [False]:
                    for global_std_mode in ["multiply"]:
                        for init_from in ["baselines"]:
                            for init_vnn_name, init_vnn_weights in [
                                ("xnfb0x2", "xavier_normal_fb:stds:0.01:0.001"),
                            ]:
                                for training_method in [
                                    "variational",
                                    "uncertainty_aware",
                                ]:
                                    for attention_filter_limit in [
                                        0.1,
                                        0.5,
                                        0.7,
                                        1,
                                        10,
                                    ]:

                                        if global_std_mode == "none":
                                            gstds = [0]
                                        else:
                                            gstds = [0.05]

                                        for gstd in gstds:

                                            name = (
                                                f"activation_{activation_mode}{'/bn' if use_batch_norm else ''}/gstd-mode_{global_std_mode}"
                                                + (
                                                    ""
                                                    if global_std_mode == "none"
                                                    else f"/gstd_{gstd}"
                                                )
                                                + f"/train_{training_method}/afl_{attention_filter_limit}"
                                            )

                                            experiments.append(
                                                (
                                                    [
                                                        "train",
                                                        name,
                                                        net,
                                                        f"vnn_{vnn_base_name(net)}/iv_{init_from}_{init_vnn_name}",
                                                    ],
                                                    {
                                                        "train_samples": samples,
                                                        "batch_norm_mode": activation_mode,
                                                        "activation_mode": activation_mode,
                                                        "use_batch_norm": use_batch_norm,
                                                        "global_std_mode": global_std_mode,
                                                        "GLOBAL_STD": gstd,
                                                        "init_vnn_from": f"models/{init_from}/{vnn_base_name(net)}",
                                                        "INIT_WEIGHTS": init_vnn_weights,
                                                        "init_vnn_from_original": init_from
                                                        == "original",
                                                        "training_method": training_method,
                                                        "attention_filter_limit": attention_filter_limit,
                                                    },
                                                )
                                            )
    random.shuffle(experiments)
    run_experiments(experiments, devices, processes_per_device, debug)


def train_vgat_for_unecrtainty_aware(
    networks=["vgat"], devices=8, processes_per_device=2, debug=False
):

    if not isinstance(devices, list):
        devices = [f"cuda:{d}" for d in range(devices)]

    experiments = []

    for net in networks:
        for samples in [7, 2]:
            for activation_mode in ["mean"]:
                for use_batch_norm in [False]:
                    for global_std_mode in ["multiply"]:
                        for init_from in ["baselines"]:
                            for init_vnn_name, init_vnn_weights in [
                                ("xnfb0x2", "xavier_normal_fb:stds:0.01:0.001"),
                            ]:
                                if global_std_mode == "none":
                                    gstds = [0]
                                else:
                                    gstds = [0.05, 0.1, 0.5, 1]

                                for gstd in gstds:

                                    name = (
                                        f"activation_{activation_mode}{'/bn' if use_batch_norm else ''}/gstd-mode_{global_std_mode}"
                                        + (
                                            ""
                                            if global_std_mode == "none"
                                            else f"/gstd_{gstd}"
                                        )
                                    )

                                    experiments.append(
                                        (
                                            [
                                                "train",
                                                name,
                                                net,
                                                f"vnn_{vnn_base_name(net)}/iv_{init_from}_{init_vnn_name}",
                                            ],
                                            {
                                                "train_samples": samples,
                                                "batch_norm_mode": activation_mode,
                                                "activation_mode": activation_mode,
                                                "use_batch_norm": use_batch_norm,
                                                "global_std_mode": global_std_mode,
                                                "GLOBAL_STD": gstd,
                                                "init_vnn_from": f"models/{init_from}/{vnn_base_name(net)}",
                                                "INIT_WEIGHTS": init_vnn_weights,
                                                "init_vnn_from_original": init_from
                                                == "original",
                                            },
                                        )
                                    )
    random.shuffle(experiments)
    run_experiments(experiments, devices, processes_per_device, debug)



def test_uncertainty_aware_from_best_vgat(
    networks=["uafmcivgat"], devices=8, processes_per_device=1, debug=False
):

    if not isinstance(devices, list):
        devices = [f"cuda:{d}" for d in range(devices)]

    experiments = []

    for net in networks:
        for samples in [2]:
            for activation_mode in ["mean"]:
                for use_batch_norm in [False]:
                    for global_std_mode in ["multiply"]:
                        for init_from in ["baselines"]:
                            for init_vnn_name, init_vnn_weights in [
                                ("xnfb0x2", "xavier_normal_fb:stds:0.01:0.001"),
                            ]:
                                for attention_filter_limit in [0.1, 0.5, 0.7, 1]:

                                    if global_std_mode == "none":
                                        gstds = [0]
                                    else:
                                        gstds = [0.05]
                                        # gstds = [0.05, 0.1, 0.5, 1]

                                    for gstd in gstds:

                                        name_for_loading = (
                                            f"activation_{activation_mode}{'/bn' if use_batch_norm else ''}/gstd-mode_{global_std_mode}"
                                            + (
                                                ""
                                                if global_std_mode == "none"
                                                else f"/gstd_{gstd}"
                                            )
                                        )
                                        name = (
                                            name_for_loading
                                            + f"/afl_{attention_filter_limit}"
                                        )

                                        experiments.append(
                                            (
                                                [
                                                    "test",
                                                    name,
                                                    net,
                                                    f"vnn_{vnn_base_name(net)}/iv_{init_from}_{init_vnn_name}",
                                                ],
                                                {
                                                    "train_samples": samples,
                                                    "batch_norm_mode": activation_mode,
                                                    "activation_mode": activation_mode,
                                                    "use_batch_norm": use_batch_norm,
                                                    "global_std_mode": global_std_mode,
                                                    "GLOBAL_STD": gstd,
                                                    # "init_vnn_from": f"models/{init_from}/{vnn_base_name(net)}",
                                                    "INIT_WEIGHTS": init_vnn_weights,
                                                    "init_vnn_from_original": init_from
                                                    == "original",
                                                    "attention_filter_limit": attention_filter_limit,
                                                    "load_model_from_architecture": "vgat",
                                                    "name_for_loading": name_for_loading,
                                                },
                                            )
                                        )
    random.shuffle(experiments)
    run_experiments(experiments, devices, processes_per_device, debug)


if __name__ == "__main__":
    Fire()

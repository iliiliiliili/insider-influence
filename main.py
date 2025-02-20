from collections import OrderedDict
import json
import os
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from earlystopping import EarlyStopping
from data_loader import create_train_valid_test_sets
from networks.gcn import BatchGCN
from networks.gat import BatchGAT
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    precision_recall_curve,
)
from statistics import stdev, mean
from fire import Fire

from networks.vnn_gat import VariationalBatchGAT, UncertaintyAwareVariationalBatchGAT
from networks.vnn_gcn import VariationalBatchGCN
from draw import draw_uncertain_attention_graphs, draw_uncertain_attentions


def train_model(
    model,
    dataloader,
    device,
    args: dict,
    patience: int,
    epochs: int,
    result_dir: str,
    verbose: bool = True,
    samples=None,
):

    extra_forward_args = {}

    if samples is not None:
        extra_forward_args["samples"] = samples

    print_every = 5
    tensorboard_logger = SummaryWriter(result_dir)
    if args["class_weight_balanced"]:
        class_weight = dataloader["test"].dataset.get_class_weight()
    else:
        class_weight = torch.ones(dataloader["test"].dataset.n_classes)

    class_weight = class_weight.to(device)
    torch.cuda.manual_seed(args["seed"])

    params = [{"params": model.layer_stack.parameters()}]

    optimizer = optim.Adagrad(params, lr=args["lr"], weight_decay=1e-3)

    train_loader = dataloader["train"]
    valid_loader = dataloader["valid"]
    test_loader = dataloader["test"]

    param_path = os.path.join(result_dir, "checkpoint.pt")

    # Defining the early stopping monitor
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, path=param_path, trace_func=print
    )

    # Loss function with class weights
    criterion = torch.nn.NLLLoss(class_weight)

    progress_bar = tqdm(total=epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = 0.0
        train_totals = 0.0

        for _, (data, target) in enumerate(train_loader):
            batch_size = data[0].size(0)

            target = target.to(device)
            data = [tensor.to(device) for tensor in data]

            optimizer.zero_grad()
            output = model(data[:2], data[-1], **extra_forward_args)
            loss_train = criterion(output, target)
            train_losses += batch_size * loss_train.item()
            train_totals += batch_size
            loss_train.backward()
            optimizer.step()

        train_loss = train_losses / train_totals
        progress_bar.update()
        # print("train loss in this epoch %f %f", epoch, train_loss)
        tensorboard_logger.add_scalar("Loss/train", train_loss, epoch)
        # =========================================================================
        #   VALIDATE MODEL
        # =========================================================================

        valid_loss, best_thr, valid_stats = evaluate(
            model, class_weight, valid_loader, device
        )
        # print(
        #     f" epoch: {epoch} train_loss: {train_loss:.5f}, "
        #     f"valid_loss: {valid_loss:.5f}, best_thr: {best_thr}"
        # )

        tensorboard_logger.add_scalar("Loss/valid", valid_loss, epoch)

        tensorboard_logger.add_scalar("Precision/raw", valid_stats["prec"][0], epoch)
        tensorboard_logger.add_scalar(
            "Precision/threshold", valid_stats["prec"][1], epoch
        )

        tensorboard_logger.add_scalar("Recall/raw", valid_stats["rec"][0], epoch)
        tensorboard_logger.add_scalar("Recall/threshold", valid_stats["rec"][1], epoch)

        tensorboard_logger.add_scalar("F1/raw", valid_stats["f1"][0], epoch)
        tensorboard_logger.add_scalar("F1/threshold", valid_stats["f1"][1], epoch)

        early_stopping(valid_loss, model)

        if epoch % print_every == 0:
            progress_bar.set_description(
                f"Epoch {epoch}/{epochs} Train Loss: {train_loss:.5f} Valid Loss: {valid_loss:.5f}"
            )

        if early_stopping.early_stop:
            if verbose:
                progress_bar.set_description("Early stopping on epoch {}".format(epoch))

            break
    # TODO: should the validation be done after the best model is loaded?
    model.load_state_dict(torch.load(param_path, weights_only=True))
    model.eval()

    _, best_thr, _ = evaluate(model, class_weight, valid_loader, device)

    print(
        f" epoch: {epoch} train_loss: {train_loss}, "
        f"valid_loss: {valid_loss}, best_thr: {best_thr}"
    )
    test_loss, _, test_stats = evaluate(
        model, class_weight, test_loader, device, best_thr=best_thr
    )
    tensorboard_logger.add_scalar("Loss/test", test_loss, epoch)
    tensorboard_logger.add_pr_curve(
        "Test", test_stats["labels"], test_stats["predictions"]
    )

    tensorboard_logger.add_hparams(
        hparam_dict={
            "lr": args["lr"],
            "batch": args["batch_size"],
            "model": args["model"],
            "hidden-units": args["hidden_units"]
            + (" heads-" + args["heads"] if args["model"] == "gat" else ""),
            "patience": patience,
            "train-size": args["train_ratio"],
            "valid-ratio": args["valid_ratio"],
            "drop-out": args["dropout"],
            "epochs": epochs,
            "last-epoch": epoch,
            "seed": args["seed"],
            "data-split-seed": args["data_split_seed"],
            "weight-decay": 1e-3,
            "class_weight_balanced": args["class_weight_balanced"],
            "best_threshold": best_thr,
        },
        metric_dict={
            "test_loss": test_loss,
            "test_auc": test_stats["auc"],
            "test_prec": test_stats["prec"][0],
            "test_rec": test_stats["rec"][0],
            "test_f1": test_stats["f1"][0],
            "test_prec_thr": test_stats["prec"][1],
            "test_rec_thr": test_stats["rec"][1],
            "test_f1_thr": test_stats["f1"][1],
        },
    )
    tensorboard_logger.close()
    return model


def calculate_metrics(y_true, y_score, y_pred, loss, total, best_thr):

    if best_thr is None:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]

    y_score = np.array(y_score)
    y_pred_thr = np.zeros_like(y_score)
    y_pred_thr[y_score > best_thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    prec_th, rec_th, f1_th, _ = precision_recall_fscore_support(
        y_true, y_pred_thr, average="binary"
    )

    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    acc_th = accuracy_score(y_true, y_pred_thr)

    return (
        loss / total,
        best_thr,
        {
            "auc": auc,
            "acc": [acc, acc_th],
            "prec": [prec, prec_th],
            "rec": [rec, rec_th],
            "f1": [f1, f1_th],
            "predictions": torch.tensor(np.exp(y_score)),
            "labels": torch.tensor(y_true),
            "predicted_labels": y_pred_thr,
        },
    )


def evaluate(model, class_weight, loader, device, best_thr=None, samples=None):

    extra_forward_args = {}

    if samples is not None:
        extra_forward_args["samples"] = samples

    model.eval()
    total = 0.0
    loss = 0.0
    y_true, y_pred, y_score = [], [], []
    class_weight = class_weight.to(device)

    for _, (data, target) in enumerate(loader):
        # graph, features, labels, vertices = batch
        bs = data[0].size(0)

        target = target.to(device)  # labels
        data = [tensor.to(device) for tensor in data]

        output = model(data[:2], data[-1], **extra_forward_args)
        loss_batch = F.nll_loss(output, target, class_weight)
        loss += bs * loss_batch.item()
        y_true += target.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        total += bs

    return calculate_metrics(y_true, y_score, y_pred, loss, total, best_thr)


def evaluate_with_uncertainty(
    model,
    class_weight,
    loader,
    device,
    best_thr=None,
    samples=None,
    draw_uncertainty_graphs=False,
):

    extra_forward_args = {}

    if samples is not None:
        extra_forward_args["samples"] = samples

    model.eval()
    total = 0.0
    loss = 0.0
    y_true, y_pred, y_score = [], [], []
    all_uncertainty_scores = []
    class_weight = class_weight.to(device)

    for _, (data, target) in enumerate(loader):
        # graph, features, labels, vertices = batch
        bs = data[0].size(0)

        target = target.to(device)  # labels
        data = [tensor.to(device) for tensor in data]

        output, uncertainty = model(
            data[:2], data[-1], return_uncertainty=True, **extra_forward_args
        )
        uncertainty_scores = (uncertainty / output.abs()).mean(axis=-1)
        loss_batch = F.nll_loss(output, target, class_weight)
        loss += bs * loss_batch.item()
        y_true += target.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        all_uncertainty_scores += uncertainty_scores.data.tolist()
        total += bs

    result = calculate_metrics(y_true, y_score, y_pred, loss, total, best_thr)

    return *result, all_uncertainty_scores


def evaluate_with_uncertainty_and_attention(
    model,
    class_weight,
    loader,
    device,
    best_thr=None,
    samples=None,
    draw_uncertainty_graphs=False,
    plots_folder_path=None,
):

    extra_forward_args = {}

    if samples is not None:
        extra_forward_args["samples"] = samples

    model.eval()
    total = 0.0
    loss = 0.0
    y_true, y_pred, y_score = [], [], []
    all_uncertainty_scores = []
    class_weight = class_weight.to(device)

    for _, (data, target) in enumerate(loader):
        # gjraph, features, labels, vertices = batch
        bs = data[0].size(0)

        target = target.to(device)  # labels
        data = [tensor.to(device) for tensor in data]

        output, uncertainty, attentions = model(
            data[:2], data[-1], return_uncertainty=True, **extra_forward_args
        )

        # draw_uncertain_attentions(attentions, plots_folder_path)
        draw_uncertain_attention_graphs(attentions, data[0], plots_folder_path)

        uncertainty_scores = (uncertainty / output.abs()).mean(axis=-1)
        loss_batch = F.nll_loss(output, target, class_weight)
        loss += bs * loss_batch.item()
        y_true += target.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        all_uncertainty_scores += uncertainty_scores.data.tolist()
        total += bs

    result = calculate_metrics(y_true, y_score, y_pred, loss, total, best_thr)

    return *result, all_uncertainty_scores


def evaluate_predictions(df):
    prec, rec, f1, _ = precision_recall_fscore_support(
        df.label, df.prediction, average="binary", zero_division=0
    )

    acc = accuracy_score(df.label, df.prediction)

    if df.label.nunique() == 2:
        auc = roc_auc_score(df.label, df.score)
    elif df.label.nunique() == 1:
        auc = None
    else:
        raise NotImplementedError("Unexpected number of different labels")
    return pd.Series(
        {
            "prec": prec,
            "rec": rec,
            "f1": f1,
            "acc": acc,
            "auc": auc,
            "n_pos_labels": df.label.sum() / df.shape[0],
            "n_samples": df.shape[0],
        }
    )


def get_parameters(
    horizon: str,
    frequency: str,
    direction: str,
    architecture: str,
    seed: Union[int, List[int]],
    name: str,
    path: str,
    dataset: str,
):

    with open("./models/gat_gcn_parameters.json", "r") as file:
        model_parameters = json.load(file)

    data_seed = seed

    if isinstance(seed, list):
        data_seed = seed[1]
        seed = seed[0]

    architecture_for_parameters = {
        "vgcn": "gcn",
        "vgat": "gat",
        "uavgat": "gat",
        "gcn": "gcn",
        "gat": "gat",
    }[architecture]

    params = model_parameters[
        f"{architecture_for_parameters}_{horizon}_{frequency}_{direction}"
    ]
    args = {
        "data_split_seed": data_seed,
        "seed": seed,
        "batch_size": int(
            model_parameters[
                f"{architecture_for_parameters}_{horizon}_{frequency}_{direction}"
            ]["batch"]
        ),
        "public_file_dir": f"./{dataset}/{horizon}_{frequency}_{direction}/",
        "shuffle": False,
        "train_ratio": 75,
        "valid_ratio": 12.5,
        "model": architecture,
        "hidden_units": params["hidden-units"],
        "heads": params["heads"],
        "dropout": params["drop-out"],
        "class_weight_balanced": True,
        "lr": params["lr"],
        "name": name,
        "path": path,
    }

    return args


def is_variational_model(architecture):
    return architecture in ["vgcn", "vgat", "uavgat"]


def main(
    mode="train",
    name="retrained",
    networks=["gat", "gcn"],
    path="baselines",
    device="cuda:0",
    results_folder="results",
    models_folder="models",
    plots_folder="plots",
    create_tables=False,
    seeds=[1, 2, 6, 5, 10, 40, 43, 46, 50],
    runs_per_variational_model=5,
    train_samples=1,
    test_samples=[2, 5, 10, 20, 40, 50],
    init_vnn_from=None,
    init_vnn_from_original=False,
    ignore_existing=False,
    test_with_uncertainty=False,
    draw_uncertainty_graphs=False,
    dataset_folder="data",
    test_model_architecture=None,
    override_horizon=None,
    override_frequency=None,
    override_direction=None,
    **vnn_kwargs,
):
    
    if not isinstance(networks, list):
        networks = [networks]

    train = (mode == "train") or (mode == True)

    if not train:
        if mode not in ("test", False):
            raise ValueError(f"Mode should be train or test (True or False)")

    datasets: List[Tuple[str, str, str, str]] = []
    for architecture in networks:
        for horizon in [override_horizon] if override_horizon else ["Lead-lag", "Simultaneous"]:
            for frequency in [override_frequency] if override_frequency else ["D", "W"]:
                for direction in [override_direction] if override_direction else ["Buy", "Sell"]:
                    datasets.append((architecture, horizon, frequency, direction))

    # Main result for best GCN and GAT architectures
    table_5_performance = pd.DataFrame(index=pd.MultiIndex.from_tuples(datasets))

    # Only samples with non-own securities
    table_8_non_own_securities = pd.DataFrame(index=pd.MultiIndex.from_tuples(datasets))

    # Only only samples with non-own securities traded by insiders themselves
    table_9_non_own_securities_self_trading = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(datasets)
    )

    prediction_list: List[pd.DataFrame] = []

    for architecture, horizon, frequency, direction in datasets:

        single_model_result = {
            "f1": [],
            "auc": [],
            "non_own_f1": [],
            "non_own_auc": [],
            "insiders_non_own_f1": [],
            "insiders_non_own_auc": [],
            "seeds": seeds,
        }

        if is_variational_model(architecture):
            runs = runs_per_variational_model
            single_model_result = [
                {
                    "f1": [],
                    "auc": [],
                    "non_own_f1": [],
                    "non_own_auc": [],
                    "insiders_non_own_f1": [],
                    "insiders_non_own_auc": [],
                    "seeds": seeds,
                }
                for _ in test_samples
            ]
            vnn_subname = f"_s{train_samples}"
        else:
            runs = 1
            vnn_subname = ""
            train_samples = None
            test_samples = None

        results_folder_path = (
            Path(results_folder)
            / path
            / name
            / f"{architecture}{vnn_subname}_{horizon}_{frequency}_{direction}"
        )

        plots_folder_path = (
            Path(plots_folder)
            / path
            / name
            / f"{architecture}{vnn_subname}_{horizon}_{frequency}_{direction}"
        )

        if (not ignore_existing) and os.path.exists(
            results_folder_path / "result.json"
        ):
            print(f"Already tested {results_folder_path}")
            break

        for sid, seed in enumerate(seeds):

            data_seed = seed

            print(
                {
                    "seed": seed,
                    "sid": f"{sid + 1} / {len(seeds)}",
                    "architecture": architecture,
                    "horizon": horizon,
                    "frequency": frequency,
                    "direction": direction,
                }
            )

            args = get_parameters(
                horizon,
                frequency,
                direction,
                architecture,
                seed,
                name,
                path,
                dataset_folder,
            )

            np.random.seed(data_seed)
            torch.manual_seed(data_seed)
            data_loader = create_train_valid_test_sets(
                args, batch_size=args["batch_size"]
            )

            np.random.seed(seed)
            torch.manual_seed(seed)

            n_neighbors = data_loader["test"].dataset.n_neighbors
            n_classes = data_loader["test"].dataset.get_num_class()

            feature_dim = data_loader["test"].dataset.get_feature_dimension()
            n_units = (
                [feature_dim]
                + [int(x) for x in args["hidden_units"].strip().split(",")]
                + [data_loader["test"].dataset.n_classes]
            )

            # Model and optimizer
            if args["model"] == "gcn":
                model = BatchGCN(
                    n_neighbors=n_neighbors,
                    n_units=n_units,
                    dropout=args["dropout"],
                )
            elif args["model"] == "vgcn":
                model = VariationalBatchGCN(
                    n_units=n_units,
                    **vnn_kwargs,
                )
            elif args["model"] == "gat":
                n_heads = [int(x) for x in args["heads"].strip().split(",")]
                model = BatchGAT(  # pretrained_emb=embedding,
                    n_units=n_units,
                    n_heads=n_heads,
                    dropout=args["dropout"],
                )
            elif args["model"] == "vgat":
                n_heads = [int(x) for x in args["heads"].strip().split(",")]
                model = VariationalBatchGAT(
                    n_units=n_units,
                    n_heads=n_heads,
                    **vnn_kwargs,
                )
            elif args["model"] == "uavgat":
                n_heads = [int(x) for x in args["heads"].strip().split(",")]
                model = UncertaintyAwareVariationalBatchGAT(
                    n_units=n_units,
                    n_heads=n_heads,
                    **vnn_kwargs,
                )
            else:
                raise NotImplementedError

            model.to(device)

            model_path = (
                Path(models_folder)
                / path
                / name
                / f"{architecture}{vnn_subname}_{horizon}_{frequency}_{direction}_seed_{seed}"
            )

            if init_vnn_from:
                print(f"Init vnn weights from {init_vnn_from}")

                weights_path = (
                    (
                        Path(init_vnn_from + f"_{horizon}_{frequency}_{direction}")
                        / "checkpoint.pt"
                    )
                    if init_vnn_from_original
                    else (
                        Path(
                            init_vnn_from
                            + f"_{horizon}_{frequency}_{direction}_seed_{seed}"
                        )
                        / "checkpoint.pt"
                    )
                )

                weights = torch.load(weights_path, weights_only=True)
                weights = OrderedDict([[k, v.to(device)] for k, v in weights.items()])

                def pair_parameter(name):
                    return (name, name.replace("means.0.", "").replace("means.", ""))

                paired_parameters = [
                    pair_parameter(a) for a in model.state_dict().keys() if "means" in a
                ]
                unpaired_parameters = [
                    a
                    for a in model.state_dict().keys()
                    if ("means" not in a) and ("stds" not in a)
                ]

                final_params = {}

                for a, b in paired_parameters:
                    final_params[a] = weights[b]

                for a in unpaired_parameters:
                    final_params[a] = weights[a]

                model.load_state_dict(final_params, strict=False)

            if train:
                os.makedirs(results_folder, exist_ok=True)

                train_model(
                    model=model,
                    dataloader=data_loader,
                    args=args,
                    device=device,
                    patience=10,
                    epochs=500,
                    result_dir=model_path,
                    samples=train_samples,
                )
            else:
                
                if test_model_architecture is None:
                    test_model_path = model_path
                else:
                    test_model_path = (
                        Path(models_folder)
                        / path
                        / name
                        / f"{test_model_architecture}{vnn_subname}_{horizon}_{frequency}_{direction}_seed_{seed}"
                    )
                    print("Loading model from", test_model_path)
                
                test_model_path = Path(test_model_path)

                path_model_checkpoint = test_model_path / "checkpoint.pt"
                model.load_state_dict(
                    torch.load(path_model_checkpoint, weights_only=True)
                )

            model.eval()

            test_loader = data_loader["test"]
            test_loader.sampler.shuffle = False
            data_loader["valid"].sampler.shuffle = False

            if args["class_weight_balanced"]:
                class_weight = test_loader.dataset.get_class_weight()
            else:
                class_weight = torch.ones(test_loader.dataset.n_classes)

            for r in range(runs):
                for i, samples in enumerate(test_samples):

                    print(
                        {
                            "rid": f"{r + 1} / {runs}",
                            "said": f"{i + 1} / {len(test_samples)}",
                        },
                        end="\r",
                    )

                    single_model_result[i]["test_samples"] = samples

                    if test_with_uncertainty:

                        if architecture in ["vgat", "uavgat"]:
                            valid_loss, best_thr, valid_stats, uncertainty_scores = (
                                evaluate_with_uncertainty_and_attention(
                                    model,
                                    class_weight,
                                    data_loader["valid"],
                                    device,
                                    samples=samples,
                                    draw_uncertainty_graphs=draw_uncertainty_graphs,
                                    plots_folder_path=plots_folder_path,
                                )
                            )
                        else:

                            valid_loss, best_thr, valid_stats, uncertainty_scores = (
                                evaluate_with_uncertainty(
                                    model,
                                    class_weight,
                                    data_loader["valid"],
                                    device,
                                    samples=samples,
                                    draw_uncertainty_graphs=draw_uncertainty_graphs,
                                    # plots_folder_path=plots_folder_path,
                                )
                            )
                    else:
                        valid_loss, best_thr, valid_stats = evaluate(
                            model,
                            class_weight,
                            data_loader["valid"],
                            device,
                            samples=samples,
                        )

                    distances = []
                    family_flags = []
                    own_company_flags = []
                    for data, _ in test_loader:
                        (
                            _,
                            _,
                            batch_distances,
                            batch_family_flags,
                            batch_own_company_flags,
                            _,
                        ) = data
                        distances.append(batch_distances.numpy())
                        family_flags.append(batch_family_flags.numpy())
                        own_company_flags.append(batch_own_company_flags.numpy())
                    distances = np.hstack(distances)
                    family_flags = np.hstack(family_flags)
                    own_company_flags = np.hstack(own_company_flags)

                    if test_with_uncertainty:
                        _, _, stats, uncertainty_scores = evaluate_with_uncertainty(
                            model, class_weight, test_loader, device, best_thr=best_thr
                        )
                    else:
                        _, _, stats = evaluate(
                            model, class_weight, test_loader, device, best_thr=best_thr
                        )

                    table_5_performance.at[
                        (architecture, horizon, frequency, direction), "F1-score"
                    ] = stats["f1"][1]
                    table_5_performance.at[
                        (architecture, horizon, frequency, direction), "AUC"
                    ] = stats["auc"]

                    single_model_result[i]["f1"].append(stats["f1"][1])
                    single_model_result[i]["auc"].append(stats["auc"])

                    predictions = pd.DataFrame(
                        [
                            own_company_flags,
                            family_flags,
                            distances,
                            stats["predicted_labels"],
                            stats["labels"].numpy(),
                            stats["predictions"].numpy(),
                        ],
                        index=[
                            "own_company_flag",
                            "family_flag",
                            "distance",
                            "prediction",
                            "label",
                            "score",
                        ],
                    ).T

                    predictions["best_thr"] = np.exp(best_thr)
                    predictions["dataset"] = f"{horizon}_{frequency}_{direction}"
                    predictions["architecture"] = architecture
                    predictions["dataset_seed"] = args["data_split_seed"]
                    predictions["seed"] = args["seed"]
                    prediction_list.append(predictions)

                    non_own_companies = evaluate_predictions(
                        predictions[predictions.own_company_flag == 0]
                    )
                    table_8_non_own_securities.at[
                        (architecture, horizon, frequency, direction), "F1-score"
                    ] = non_own_companies.f1
                    table_8_non_own_securities.at[
                        (architecture, horizon, frequency, direction), "AUC"
                    ] = non_own_companies.auc

                    single_model_result[i]["non_own_f1"].append(non_own_companies.f1)
                    single_model_result[i]["non_own_auc"].append(non_own_companies.auc)

                    insiders_non_own_companies = evaluate_predictions(
                        predictions[
                            (predictions.own_company_flag == 0)
                            & (predictions.family_flag == 0)
                        ]
                    )
                    table_9_non_own_securities_self_trading.at[
                        (architecture, horizon, frequency, direction), "F1-score"
                    ] = insiders_non_own_companies.f1
                    table_9_non_own_securities_self_trading.at[
                        (architecture, horizon, frequency, direction), "AUC"
                    ] = insiders_non_own_companies.auc

                    single_model_result[i]["insiders_non_own_f1"].append(
                        non_own_companies.f1
                    )
                    single_model_result[i]["insiders_non_own_auc"].append(
                        non_own_companies.auc
                    )

        if is_variational_model(architecture):
            for i in range(len(single_model_result)):
                keys = [*single_model_result[i].keys()]
                for key in keys:
                    if key not in ["seeds", "test_samples"]:
                        std = stdev(single_model_result[i][key])
                        single_model_result[i][key] = mean(single_model_result[i][key])
                        single_model_result[i][key + "_std"] = std

        else:
            keys = [*single_model_result.keys()]
            for key in keys:
                if key != "seeds":
                    std = stdev(single_model_result[key])
                    single_model_result[key] = mean(single_model_result[key])
                    single_model_result[key + "_std"] = std

        results_folder_path = (
            Path(results_folder)
            / path
            / name
            / f"{architecture}{vnn_subname}_{horizon}_{frequency}_{direction}"
        )
        os.makedirs(results_folder_path, exist_ok=True)

        with open(results_folder_path / "result.json", "w") as f:
            json.dump(single_model_result, f)

    if len(prediction_list) <= 0:
        print("No predictions to evaluate")
        return

    combined_predictions = pd.concat(prediction_list, axis=0)
    combined_predictions = combined_predictions[
        combined_predictions.architecture == "gat"
    ]

    if create_tables:

        # Performances for samples with different distances between the traded and company
        table_10 = pd.DataFrame(
            index=pd.MultiIndex.from_product([["insider", "family"], range(4)])
        )

        for family_flag, investor_type in enumerate(["insider", "family"]):
            for distance in range(5):
                if family_flag:
                    if distance == 1:
                        continue
                    elif distance > 1:
                        adj_distance = distance - 1
                    else:
                        adj_distance = distance
                else:
                    if distance < 4:
                        adj_distance = distance
                    else:
                        continue
                distance_perofmance = evaluate_predictions(
                    combined_predictions[
                        (combined_predictions.distance == distance)
                        & (combined_predictions.family_flag == family_flag)
                    ]
                )
                table_10.at[(investor_type, adj_distance), "F1-score"] = (
                    distance_perofmance.f1
                )
                table_10.at[(investor_type, adj_distance), "AUC"] = (
                    distance_perofmance.auc
                )

        table_5_performance = (
            table_5_performance.round(2)
            .unstack([1, 2, 3])
            .stack(0, future_stack=True)
            .sort_index(ascending=[True, False])
        )
        print("TABLE 5:\n", table_5_performance)

        table_8_non_own_securities = (
            table_8_non_own_securities.round(2)
            .unstack([1, 2, 3])
            .stack(0, future_stack=True)
            .sort_index(ascending=[True, False])
        )
        print("TABLE 8:\n", table_8_non_own_securities)

        table_9_non_own_securities_self_trading = (
            table_9_non_own_securities_self_trading.round(2)
            .unstack([1, 2, 3])
            .stack(0, future_stack=True)
            .sort_index(ascending=[True, False])
        )
        print("TABLE 9:\n", table_9_non_own_securities_self_trading)

        table_10 = (
            table_10.round(2)
            .unstack(1)
            .stack(0, future_stack=True)
            .sort_index(ascending=[False, False])
        )
        print("TABLE 10:\n", table_10)

        results = {
            "table_5": table_5_performance,
            "table_8": table_8_non_own_securities,
            "table_9": table_9_non_own_securities_self_trading,
            "table_10": table_10,
        }

        if results_folder is not None:
            results_folder_path = Path(results_folder) / path / name
            os.makedirs(results_folder_path, exist_ok=True)

            with open(results_folder_path / ".gitignore", "w") as f:
                f.write("*\n")

            for name, result in results.items():
                result.to_csv(results_folder_path / (name + ".csv"))


if __name__ == "__main__":
    Fire(main)

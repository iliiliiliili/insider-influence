from fire import Fire
from main import main
from multiprocessing import Pool


def vnn_base_name(vnn_name):
    if "gat" in vnn_name:
        return "gat"
    if "gcn" in vnn_name:
        return "gcn"

    return vnn_name


def run_experiments(experiments, devices, processes_per_device):

    experiments_per_device = [0 for _ in devices]

    experiments_to_assign = len(experiments)

    for i in range(len(experiments_per_device)):
        experiments_per_device[i] = min(experiments_to_assign, processes_per_device)
        experiments_to_assign -= experiments_per_device[i]

    while (experiments_to_assign > 0):
        for i in range(len(experiments_per_device)):
            experiments_per_device[i] += 1
            experiments_to_assign -= 1

            if experiments_to_assign <= 0:
                break

    pools = [Pool(processes_per_device) for _ in devices]

    for exp_count, device, pool in zip(experiments_per_device, devices, pools):

        for i in range(exp_count):
            experiment_args, experiment_kwargs = experiments.pop(0)
            print(experiment_args, experiment_kwargs)
            pool.apply_async(main, args=(*experiment_args, ), kwds={"device":device, **experiment_kwargs})

    for pool in pools:
        pool.close()
        pool.join()

    print("Done")


def vnn(networks="vgat", devices=8, processes_per_device=3):

    if not isinstance(devices, list):
        devices = [f"cuda:{d}" for d in range(devices)]

    experiments = []

    for net in networks:
        for samples in range(2, 10):
            experiments.append((["train", f"n", net, f"vnn_{vnn_base_name(net)}"], {"train_samples": samples}))

    run_experiments(experiments, devices, processes_per_device)
        


if __name__ == "__main__":
    Fire()

from fire import Fire
from main import main
from multiprocessing import Pool


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


def seeds(networks="gat", devices=8, processes_per_device=3):

    if not isinstance(devices, list):
        devices = [f"cuda:{d}" for d in range(devices)]

    experiments = []

    for seed in range(20, 50):
        for data_seed in range(20, 50):
            experiments.append((["train", f"seeds/{seed}_{data_seed}", networks], {"seed":[seed, data_seed]}))
    
    run_experiments(experiments, devices, processes_per_device)
        


if __name__ == "__main__":
    Fire()

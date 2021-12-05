import argparse
import json
from datetime import datetime
import platform
from pathlib import Path
from time import time
from typing import Callable, Iterable
import scipy.stats as st

import numpy
import numpy as np
from numpy import mean, std
from pandas import DataFrame

from deepstochlog.trainer import PandasLogger

eval_root = Path(__file__).parent


def create_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("-r", "--runs", type=int, default=5)
    parser.add_argument("-t", "--only_time", default=False, action="store_true")
    return parser


def mean_confidence_interval(data: numpy.array, confidence=0.95):
    return st.t.interval(
        confidence, len(data) - 1, loc=np.mean(data), scale=st.sem(data)
    )


def pick_test_accuracy(
    df: DataFrame, maximize_attr: Iterable[str], target_attr: Iterable[str]
):
    new_df = df
    for attr in maximize_attr:
        new_df = new_df[new_df[attr].values == new_df[attr].values.max()]
    return [new_df[attr].iloc[-1] for attr in target_attr]


def evaluate_deepstochlog_task(
    runner: Callable,
    arguments,
    runs=5,
    name=None,
    only_time=False,
    maximize_attr=("Val acc", "Val P(cor)"),
    target_attr=("Test acc", "Test P(cor)"),
):
    if only_time:
        print("Not measuring performance, only time to run.")
        maximize_attr = ("time",)
        target_attr = ("time",)
        arguments["val_size"] = 0
        arguments["test_size"] = 0
        arguments["log_freq"] = 100000
        if "val_number_digits" in arguments:
            arguments["val_number_digits"] = 0

    cur_time = str(time())
    run_name = cur_time[: cur_time.index(".")]
    host_name = platform.node()
    folder = (
        eval_root
        / ".."
        / "data"
        / "eval"
        / (
            (("time-" if only_time else "") + name + "-" if name is not None else "")
            + run_name
            + "-"
            + host_name
        )
    )
    folder.mkdir(exist_ok=True, parents=True)

    test_accs = np.zeros((runs, len(target_attr)))
    outputs = []

    for run in range(runs):
        logger = PandasLogger()

        output = runner(logger=logger, **arguments)
        outputs.append(output)
        if output and isinstance(output, dict) and "model" in output:
            del output["model"]

        df = logger.df
        print(df)

        test_acc = pick_test_accuracy(
            df=df, maximize_attr=maximize_attr, target_attr=target_attr
        )
        for j in range(len(target_attr)):
            test_accs[run, j] = test_acc[j]

        # save the df
        csv = df.to_csv(index=False)
        with open(folder / "{}.csv".format(run), "w") as csv_file:
            csv_file.write(csv)

    results = {
        "host_name": host_name,
        "run_name": run_name,
        "date": str(datetime.now()),
        "runs": runs,
        "maximize_attr": maximize_attr,
        "target_attr": target_attr,
        "test_accs": test_accs.tolist(),
        "means": mean(test_accs, axis=0).tolist(),
        "standard deviatations": std(test_accs, axis=0).tolist(),
        "95% interval": [
            mean_confidence_interval(test_accs[:, i], confidence=0.95)
            for i in range(len(target_attr))
        ],
        "arguments": arguments,
        "outputs": outputs,
    }

    results_json = json.dumps(results, indent=4)

    print("Results:", results_json)
    with open(folder / "results.csv", "w") as results_file:
        results_file.write(results_json)

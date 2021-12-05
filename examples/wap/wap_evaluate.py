import sys
from pathlib import Path

from examples.wap import wap
from examples.evaluate import evaluate_deepstochlog_task


eval_root = Path(__file__).parent


def main():
    arguments = {
        "train_size": None,
        "val_size": None,
        "test_size": None,
        "epochs": 40,
        "log_freq": 30,
        "test_example_idx": [],
        "verbose": False,
    }

    evaluate_deepstochlog_task(
        runner=wap.run,
        arguments=arguments,
        runs=5,
        name="wap",
        maximize_attr=("Val acc", "Val P(cor)"),
        target_attr=("Test acc", "Test P(cor)", "time"),
    )


if __name__ == "__main__":
    main()

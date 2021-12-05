import sys
from pathlib import Path

from examples.bracket import bracket
from examples.evaluate import evaluate_deepstochlog_task


eval_root = Path(__file__).parent


def main(max_length: int = 10):
    print("Running for max_length", max_length)
    arguments = {
        "min_length": 2,
        "max_length": max_length,
        "train_size": 1000,
        "val_size": 200,
        "test_size": 200,
        "batch_size": 4,
        "val_num_digits": 1000,
        "allow_uneven": False,
        "only_bracket_language_examples": True,
        "seed": 42,
        "set_program_seed": False,
        "epochs": 1,
        "log_freq": 5,
        "test_example_idx": [],
        "verbose": False,
    }

    evaluate_deepstochlog_task(
        runner=bracket.run,
        arguments=arguments,
        runs=5,
        name=("bracket" + str(max_length)),
        maximize_attr=("Val parse acc",),
        target_attr=("Test parse acc", "time"),
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the max bracket length")

    main(max_length=int(sys.argv[1]))

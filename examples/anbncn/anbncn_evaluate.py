from pathlib import Path

from examples.anbncn import anbncn
from examples.evaluate import evaluate_deepstochlog_task, create_default_parser

eval_root = Path(__file__).parent


def main(max_length: int = 12, epochs: int = 1, runs: int = 5, only_time: bool = False):

    print(
        "Running {} runs of {} epochs each for max length {}".format(
            runs, epochs, max_length
        )
    )
    arguments = {
        "min_length": 3,
        "max_length": max_length,
        "train_size": 4000,
        "batch_size": 4,
        "val_size": 100,
        "test_size": 200,
        "val_num_digits": 300,
        "allow_non_threefold": False,
        "seed": 42,
        "set_program_seed": False,
        #
        "epochs": epochs,
        "log_freq": 50,
        "test_example_idx": [],
        "verbose": False,
    }

    evaluate_deepstochlog_task(
        runner=anbncn.run,
        arguments=arguments,
        runs=runs,
        name=("anbcn" + str(max_length)),
        only_time=only_time,
        maximize_attr=(
            "Val acc",
            "Val P(cor)",
            # "Val parse acc",
        ),
        target_attr=("Test acc", "Test P(cor)",
                     # "Test parse acc",
                     "time"),
    )


parser = create_default_parser()
parser.add_argument("-l", "--max_length", type=int, default=12)

if __name__ == "__main__":
    args = parser.parse_args()
    main(
        max_length=args.max_length,
        epochs=args.epochs,
        runs=args.runs,
        only_time=args.only_time,
    )

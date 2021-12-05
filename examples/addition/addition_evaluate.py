from pathlib import Path

from examples.addition import addition
from examples.evaluate import evaluate_deepstochlog_task, create_default_parser

eval_root = Path(__file__).parent


def main(
    digit_length: int = 1,
    epochs: int = 5,
    runs: int = 5,
    only_time: bool = False,
    greedy: bool = False,
):

    print(
        "Running {} runs of {} epochs each for digit length {}".format(
            runs, epochs, digit_length
        )
    )
    if greedy:
        print("Greedy")

    arguments = {
        "digit_length": digit_length,
        "epochs": epochs,
        "val_size": 500,
        "test_size": None,
        "log_freq": 100,
        "test_example_idx": [],
        # "seed": 42,
        "verbose": False,
        "greedy": greedy,
    }

    evaluate_deepstochlog_task(
        runner=addition.run,
        arguments=arguments,
        runs=runs,
        only_time=only_time,
        name=("addition" + str(digit_length)),
        maximize_attr=("Val acc", "Val P(cor)") if not greedy else ("Val acc",),
        target_attr=("Test acc", "Test P(cor)", "time")
        if not greedy
        else ("Test acc",),
    )


parser = create_default_parser()
parser.add_argument("-d", "--digit_length", type=int, default=1)
parser.add_argument("-g", "--greedy", default=False, action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    main(
        digit_length=args.digit_length,
        epochs=args.epochs,
        runs=args.runs,
        only_time=args.only_time,
        greedy=args.greedy,
    )

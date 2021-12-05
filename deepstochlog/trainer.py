from time import time
from typing import Callable, Tuple, List, Union

import pandas as pd
import torch
from pandas import DataFrame

from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel


class DeepStochLogLogger:
    def log_header(self, accuracy_tester_header):
        pass

    def log(
        self,
        epoch: int,
        batch_idx: int,
        total_loss: float,
        instances_since_last_log: int,
        accuracy_tester: str,
    ):
        raise NotImplementedError()

    def print(
        self,
        line: str,
    ):
        print(line)


class PrintLogger(DeepStochLogLogger):
    def log_header(self, accuracy_tester_header):
        print("Epoch\tBatch\tLoss\t\t" + accuracy_tester_header)

    def log(
        self,
        epoch: int,
        batch_idx: int,
        total_loss: float,
        instances_since_last_log: int,
        accuracy_tester: str,
    ):
        print(
            "{:>5}\t{:>5}\t{:.5f}\t\t{}".format(
                epoch,
                batch_idx,
                float(total_loss) / instances_since_last_log,
                accuracy_tester,
            )
        )


class PrintFileLogger(DeepStochLogLogger):
    def __init__(self, filepath):
        self.filepath = filepath

    def _write_and_print(self, line):
        with open(self.filepath, "a") as f:
            f.write(line)
            f.write("\n")
        print(line)

    def log_header(self, accuracy_tester_header):
        line = "Epoch\tBatch\tLoss\t\t" + accuracy_tester_header
        self._write_and_print(line)

    def print(self, line):
        self._write_and_print(line)

    def log(
        self,
        epoch: int,
        batch_idx: int,
        total_loss: float,
        instances_since_last_log: int,
        accuracy_tester: str,
    ):

        line = "{:>5}\t{:>5}\t{:.5f}\t\t{}".format(
            epoch,
            batch_idx,
            float(total_loss) / instances_since_last_log,
            accuracy_tester,
        )
        self._write_and_print(line)


class PandasLogger(DeepStochLogLogger):
    def __init__(self):
        self.df: DataFrame = DataFrame()

    def log_header(self, accuracy_tester_header):
        columns = ["Epoch", "Batch", "Loss"] + accuracy_tester_header.split("\t")
        self.df = DataFrame(data=[], columns=columns)

    def log(
        self,
        epoch: int,
        batch_idx: int,
        total_loss: float,
        instances_since_last_log: int,
        accuracy_tester: str,
    ):
        to_append: List[Union[str, float, int]] = [
            epoch,
            batch_idx,
            total_loss / instances_since_last_log,
        ]
        to_append.extend(
            [float(el) for el in accuracy_tester.split("\t") if len(el.strip()) > 0]
        )
        series = pd.Series(to_append, index=self.df.columns)
        self.df = self.df.append(series, ignore_index=True)

    def get_last_result(self):
        return self.df.iloc[[-1]]


print_logger = PrintLogger()


class DeepStochLogTrainer:
    def __init__(
        self,
        logger: DeepStochLogLogger = print_logger,
        log_freq: int = 50,
        accuracy_tester: Tuple[str, Callable[[], str]] = (),
        test_query=None,
        print_time=False,
        allow_zero_probability_examples=False,
    ):
        self.logger = logger
        self.log_freq = log_freq
        self.accuracy_tester_header, self.accuracy_tester_fn = accuracy_tester
        self.test_query = test_query
        self.print_time = print_time
        self.allow_zero_probability_examples = allow_zero_probability_examples

    def train(
        self,
        model: DeepStochLogModel,
        optimizer,
        dataloader: DataLoader,
        epochs: int,
        epsilon=1e-8,
    ):
        # Test the performance using the test query
        if self.test_query is not None:
            self.test_query()

        # Log time
        training_start = time()

        # Start training
        batch_idx = 0
        total_loss = 0
        instances_since_last_log = 0
        self.logger.log_header(self.accuracy_tester_header)
        for epoch in range(epochs):
            for batch in dataloader:

                # Cross-Entropy (CE) loss
                probabilities = model.predict_sum_product(batch)
                if self.allow_zero_probability_examples:
                    targets = torch.as_tensor(
                        [el.probability for el in batch], device=model.device
                    )
                    losses = -(
                        targets * torch.log(probabilities + epsilon)
                        + (1.0 - targets) * torch.log(1.0 - probabilities + epsilon)
                    )
                else:
                    losses = -torch.log(probabilities + epsilon)

                loss = torch.mean(losses)
                loss.backward()

                # Step optimizer for learning
                optimizer.step()
                optimizer.zero_grad()

                # Save loss for printing
                total_loss += float(loss)
                instances_since_last_log += len(batch)

                # Print the loss
                if self.should_log(batch_idx, dataloader, epoch, epochs):
                    self.logger.log(
                        epoch,
                        batch_idx,
                        total_loss,
                        instances_since_last_log,
                        self.accuracy_tester_fn(),
                    )
                    total_loss = 0
                    instances_since_last_log = 0

                batch_idx += 1

        end_time = time() - training_start
        if self.print_time:
            self.logger.print(
                "\nTraining {} epoch (totalling {} batches of size {}) took {:.2f} seconds".format(
                    epochs, epochs * len(dataloader), dataloader.batch_size, end_time
                )
            )

        # Test the performance on the first test query again
        if self.test_query is not None:
            self.test_query()

        return end_time

    def should_log(self, batch_idx, dataloader, epoch, epochs):
        return batch_idx % self.log_freq == 0 or (
            epoch == epochs - 1 and batch_idx % len(dataloader) == len(dataloader) - 1
        )

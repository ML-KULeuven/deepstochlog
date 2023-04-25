# DeepStochLog

DeepStochLog is a neuro-symbolic framework that combines grammars, logic, probabilities and neural networks.
By writing a DeepStochLog program, one can train a neural network with the given background knowledge.
One can express symbolic information about subsymbolic data in DeepStochLog and help train neural networks more efficiently this way.
For example, if the training data is made up of handwritten digit images, and we know the sum of these digits but not the individual numbers, one can express this relation in DeepStochLog and train the neural networks much faster.

DeepStochLog uses a stochastic logic approach to encoding the probabilistic logic, and is thus faster than and can deal with longer inputs than its sibling DeepProbLog in our experiments.

## Installation

### Installing SWI Prolog
DeepStochLog requires [SWI Prolog](https://www.swi-prolog.org/build/PPA.html) to run.
Run the following commands to install:
```
sudo apt-add-repository ppa:swi-prolog/stable
sudo apt-get update
sudo apt-get install swi-prolog
```

### Installing DeepStochLog package
To install DeepStochLog itself, run the following command:

```
pip install deepstochlog
```

## Running the examples

### Local dependencies

To see DeepStochLog in action, please first install [SWI Prolog](https://www.swi-prolog.org/build/PPA.html) (as explained about),
as well as the requirements listed in `requirements.txt`
```
pip install -r requirements.txt
```

### Datasets
The datasets used in the tasks used to evaluate DeepStochLog can be found in our [initial release](https://github.com/ML-KULeuven/deepstochlog/releases/tag/0.0.1).

### Addition example

To see DeepStochLog in action, navigate to `examples/addition` and run `addition.py`.

The neural definite clause grammar specification is provided in `addition.pl`.
The `addition(N)` predicate specifies/recognises that two handwritten digits *N1* and *N2* sum to *N*.
The neural probability `nn(number, [X], Y, digit)` makes the neural network with name `number` (a MNIST classifier) label input image X with the digit Y.



## Credits & Paper citation

If use this work in an academic context, please consider citing [the following paper](https://ojs.aaai.org/index.php/AAAI/article/view/21248):

```
@inproceedings{winters2022deepstochlog,
  title={Deepstochlog: Neural stochastic logic programming},
  author={Winters, Thomas and Marra, Giuseppe and Manhaeve, Robin and De Raedt, Luc},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={9},
  pages={10090--10100},
  year={2022}
}
```

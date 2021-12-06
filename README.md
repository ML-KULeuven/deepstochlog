# DeepStochLog

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
To install DeepStochLog itself, you can download this repository and run the following command:
```
python setup.py install
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

If use this work in an academic context, please consider citing [the following paper](https://arxiv.org/abs/2106.12574):

The paper is also accepted to [AAAI22](https://aaai.org/Conferences/AAAI-22/).
Please cite that version of the paper when the proceedings are out.

```
@article{winters2021deepstochlog,
  title={Deepstochlog: Neural stochastic logic programming},
  author={Winters, Thomas and Marra, Giuseppe and Manhaeve, Robin and De Raedt, Luc},
  journal={arXiv preprint arXiv:2106.12574},
  year={2021}
}
```

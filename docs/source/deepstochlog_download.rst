Download
===============================================

Installation
------------
DeepStochLog requires SWI Prolog to run. Run the following commands to install:

.. code-block:: bash

    sudo apt-add-repository ppa:swi-prolog/stable
    sudo apt-get update
    sudo apt-get install swi-prolog

To install DeepStochLog itself, run the following command:

.. code-block:: bash

    pip install deepstochlog


Requirements
------------

DeepStochLog has the following requirements:

* torch~=1.5.1
* torchvision~=0.6.1
* numpy~=1.18.1
* pandas~=1.2.4
* pyparsing~=2.4.7
* dgl~=0.6.1


Running the examples
--------------------
**Local dependencies**
To see DeepStochLog in action, please first install SWI Prolog (as explained about), as well as all requirements.

**Datasets**
The datasets used in the tasks used to evaluate DeepStochLog can be found `here <https://github.com/ML-KULeuven/deepstochlog/releases/tag/0.0.1>`_.

**Addition example**
To see DeepStochLog in action, navigate to ``examples/addition`` and run ``addition.py`` in the `github repo <https://github.com/ML-KULeuven/deepstochlog/tree/main>`_.
The neural definite clause grammar specification is provided in ``addition.pl``.
The ``addition(N)`` predicate specifies/recognises that two handwritten digits *N1* and *N2* sum to *N*.
The neural probability ``nn(number, [X], Y, digit)`` makes the neural network with name ``number`` (a MNIST classifier) label input image X with the digit Y.

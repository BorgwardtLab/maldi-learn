# `maldi-learn`

`maldi-learn` is a Python library for MALDI-TOF preprocessing and
machine learning analysis, developed by members of the [Machine Learning
and Computational Biology Lab](https://bsse.ethz.ch/mlcb) of [Prof. Dr.
Karsten Borgwardt](https://bsse.ethz.ch/mlcb/karsten.html). 

This library is a **work in progress**. Below, we provide some
information for loading &ldquo;DRIAMS&rdquo;, a large-scale data set of
mass spectra and performing some simple machine learning tasks with it.
Stay tuned for more information!

## Installation

The installation of this package requires [`poetry`](https://python-poetry.org/docs/), a 
Python package handler. This is only required if you want to try out the
code for yourself or even contribute to `maldi-learn`. In cases where
`maldi-learn` is used as a dependency, there is no need for you to do
anything.

To set up the development environment, run `poetry install` in the
project root.  To run commands in the associated virtual environment of this
package, run `poetry shell` to spawn a shell.

### Handling multiple Python versions

This project requires at least python version `3.7`.  If you want to
contribute to `maldi-learn`, we would recommend a development setup
in which an appropriate Python version is installed using
[`pyenv`](https://github.com/pyenv/pyenv), and then marking this folder
for usage with this version:

```bash
$ pyenv install 3.7.4  # Install python 3.7.4 using pyenv
$ pyenv local 3.7.4    # Mark python version 3.7.4 for usage in this folder
$ poetry install       # Setup the virtual environment for development
```

## Examples

The folder `examples` contains scripts that demonstrate a part of the
API. We recommend to start with
[`classification.py`](examples/classification.py). This examples
demonstrates a full, albeit simplified example, which we will
subsequently discuss here.

We first import all required packages. Notice that we use `dotenv`,
a package for handling environment variables in `.env` files. This makes
it very easy to specify the root folder of the data set. If you
downloaded the data set, you can add the following line to a `.env` file
in the root folder of this repository:

```
DRIAMS_ROOT=/path/to/your/download
```

The code below automatically looks for this file and stores its value in
the variable `DRIAMS_ROOT`. Alternatively, you can also set an
environment variable using `export DRIAMS_ROOT=/path/to/your/download`.

```python
import dotenv
import os

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder
from maldi_learn.driams import load_driams_dataset

from maldi_learn.utilities import stratify_by_species_and_label
from maldi_learn.vectorization import BinningVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')
```

Moving on, we perform some book-keeping and loading in the code:

```python

explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)

driams_dataset = load_driams_dataset(
            explorer.root,
            'DRIAMS-A',
            ['2015', '2017'],
            'Staphylococcus aureus',
            ['Ciprofloxacin', 'Penicillin'],
            handle_missing_resistance_measurements='remove_if_all_missing',
)
```

As you can see, the data set will contain spectra from 2015 and 2017,
from the `DRIAMS-A` site, with a species of *Staphylococcus aureus*, and
the two antibiotics ciprofloxacin and penicillin. The most interesting
part of this call is the parameter for `handle_missing_resistance_measurements`.
With its current value, a sample will be removed if there are *no*
resistance measurements available. Other values are
`remove_if_any_missing`, meaning that a *single* missing value already
triggers removal, or `keep`, meaning that *no* removal will be
performed.

Next we continue with binning the spectra. For this example, we only use
100 bins. Afterwards, we prepare a stratified split based on
ciprofloxacin&nbsp;(thus ensuring that the prevalence of the resistance
values is the same; strictly speaking this is not necessary here as we
only loaded one species, but this example is supposed to be somewhat
*instructive*).

```python
# bin spectra
bv = BinningVectorizer(100, min_bin=2000, max_bin=20000)
X = bv.fit_transform(driams_dataset.X)

# train-test split
index_train, index_test = stratify_by_species_and_label(
    driams_dataset.y, antibiotic='Ciprofloxacin'
)
```

Next, we create&nbsp;(binary) labels and train a standard logistic
regression classifier. In a real application, we would also be
performing a hyperparameter search, but for this example, a single
demonstration based on our train&ndash;test split is sufficient.

```python
y = driams_dataset.to_numpy('Ciprofloxacin')

lr = LogisticRegression()
lr.fit(X[index_train], y[index_train])
y_pred = lr.predict(X[index_test])

print(accuracy_score(y_pred, y[index_test]))
```

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

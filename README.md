# maldi-learn
Software library for MALDI-TOF preprocessing and machine learning analysis.

## Installation - development

The installation of this package requires
[poetry](https://python-poetry.org/docs/).

In order to set up a development environment run `poetry install` in the
project root.  To run commands in the associated virtual environment of this
package run `poetry shell` to spawn a shell.


### Python version

This project requires at least python version `3.7`.  In a development setup it
is recommended to install a appropriate python version using
[pyenv](https://github.com/pyenv/pyenv), and then marking this folder for usage
with this version:

```bash
 $ pyenv install 3.7.4  # Install python 3.7.4 using pyenv
 $ pyenv local 3.7.4    # Mark python version 3.7.4 for usage in this folder
 $ poetry install       # Setup the virtual environment for development
```

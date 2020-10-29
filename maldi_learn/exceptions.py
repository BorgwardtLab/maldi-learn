"""Custom exceptions for `maldi-learn`."""

import warnings


class AntibioticNotFoundException(ValueError):
    """Exception to raise if an antibiotic was not found."""

    def __init__(self, message):
        super().__init__(message)


class AntibioticNotFoundWarning(UserWarning):
    """Warning to show if an antibiotic was not found."""


class SpeciesNotFoundException(ValueError):
    """Exception to raise if a species was not found."""

    def __init__(self, message):
        super().__init__(message)


class SpeciesNotFoundWarning(UserWarning):
    """Warning to show if a species was not found."""


class SpectraNotFoundException(ValueError):
    """Exception to raise if a spectra was not found."""

    def __init__(self, message):
        super().__init__(message)


class SpectraNotFoundWarning(UserWarning):
    """Warning to show if a spectra was not found."""


def _raise_or_warn(exception, warning, message, behaviour):
    """Raise exception or show warning.

    Raises a pre-defined exception or shows a pre-defined warning,
    depending on the desired behaviour. This is a utility function that
    should be used within `maldi-learn` functions to signal problems to
    clients and users.

    Parameters
    ----------
    exception
        Exception to raise if `behaviour` is 'warn' or 'warning'. Should
        be a class name.

    warning
        Warning to show if `behaviour` is 'warn' or 'warning'. Should be
        a class name.

    message : str
        Message to use when raising an exception or showing a warning.

    behaviour : str
        Either one of 'raise', 'warn', or 'warning', to indicate the
        desired behaviour, i.e. whether to raise an exception or not.
    """

    assert behaviour in ['raise', 'warn', 'warning']

    if behaviour == 'raise':
        raise exception(message=message)
    else:
        warnings.warn(message, category=warning)

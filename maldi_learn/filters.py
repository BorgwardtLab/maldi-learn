"""Filter classes for `maldi-learn`."""

import dateparser


class DRIAMSFilter:
    """Generic filter class for spectra."""

    def __init__(self, filters=[]):
        self.filters = filters

    def __call__(self, row):
        result = True
        for filter_fn in self.filters:
            result = result & filter_fn(row)

        return result


class DRIAMSDateRangeFilter:
    def __init__(self, date_from, date_to, date_col='acquisition_date'):
        self.date_from = dateparser.parse(date_from)
        self.date_to = dateparser.parse(date_to)
        self.date_col = date_col

        assert self.date_to is not None
        assert self.date_from is not None

    def __call__(self, row):
        date = dateparser.parse(row[self.date_col])

        assert date is not None

        return self.date_to <= date <= self.date_from


class DRIAMSDateFilter:
    def __init__(self, date, date_col='acquisition_date'):
        self.date = dateparser.parse(
            date,
            settings={
                'PREFER_DAY_OF_MONTH': 'last'
            }
        )
        self.date_col = date_col

        assert self.date is not None

    def __call__(self, row):
        date = dateparser.parse(row[self.date_col])

        assert date is not None

        raise NotImplementedError('Not yet implemented')


class DRIAMSSpeciesFilter:
    def __init__(self, species=[]):
        if type(species) is not list:
            self.species = [species]
        else:
            self.species = species

    def __call__(self, row):
        for species in self.species:
            if species in row['species']:
                return True

        return False


def filter_by_machine_type(row):
    return 'MALDI1' in row['code']




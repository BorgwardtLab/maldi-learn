DRIAMS
  data
    [site]
      fid
        [year]
      raw
        [year]
      preprocessed
        [year]
      id
        [year]

DRIAMSDataset(root='...')
  - parse file tree
  - grab all information
  - do some consistency checks
  - which antibiotics?
  - which years?
  - which sites?

DRIAMSDatasetLoader(
  root='...',
  site='...',
  year='...',
  species='...',
  antibiotic='...',
  handle_missing_values='...'
)

  Returns:
    DataSet
      - X, y, ...
      - property: is_multitask

handle_missing_resistance_measurements:
  ['remove_all_missing', 'remove_any_missing', 'keep']
    'remove_all_missing': single task: remove all samples for which measurements do not exist
                          multi-task: remove all samples for which no measurements exists
    'remove_any_missing': single task: same as 'remove_all_missing'
                          multi-task: remove all samples for which at least one measurement is missing
    'keep': do not remove anything

LabelMapper:
  - maps label to value; by default: map resistant/intermediate to zero, map susceptible to 1, map remainder to NaN
  - y
  - returns y
  - algorithm:
      take mappings from dict; label that does not satisfy the mapping
      will be mapped to NaN.

DRIAMSLabelMapper

Generic point: everything that contains 'DRIAMS' in the class name is
specific to this data set; more 'general' classes exist for handling
other data sources.

---

Stratifier (?)
train_test_split (?)

---

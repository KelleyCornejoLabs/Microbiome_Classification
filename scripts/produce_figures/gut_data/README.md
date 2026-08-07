# What are the files?

`processed_truth.csv` contains the processed 'ground truth' data that Enterotyper was originally trained from
Files called `processed_truth_80/60_train/test/validation.csv` are the result of make test train script
`human_micro_comendium.csv` is as the name suggests, it is formatted. No truth labels
The models in `models` contain the models trained on the split `processed_truth` data
`entero_unlabeled` are the files used to run enterotyper
`entero_classifications` are the files produced by enterotyper containing classifications
`enterotyper_classified.csv` is the compiled dataframe with all labeles
`entero_classified_80/60.csv` is the files contianing the entries from enterotyper\_classified that correspond to the 80/60 validaiton sets


How microbiome compendium names were processed:
```python
def rename(x):
	if x == 'sample' or x == 'cluster': return x
	parts = x.replace('_', '-').split('.',5)
	if len(parts_ == 6: return parts[5]
	else: return parts[-1]
```
Applied to every columns of the microbiome compendium data

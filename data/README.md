# Dataset instructions

The public HR Analytics source CSV is **not republished in this repository** because its redistribution terms have not been verified.

1. Obtain the source dataset (`HR_comma_sep.csv`) from:
   https://github.com/RuchitaGarde/Human-Resources-Analytics
2. Check that the dataset is permitted for your intended use and keep a record of the source/version.
3. Run:

```bash
python src/prepare_dataset.py --source path/to/HR_comma_sep.csv --output data/Employee_Attrition_10000_Training_Dataset.xlsx
```

This creates 10,000 unique predictor records, with the project target column `Attrition` (Yes/No). It applies a deterministic, class-stratified sample from the first 11,991 unique source predictor rows with a fixed xorshift32 seed of `20261111`.

Original records: 14,999; removed duplicate predictor patterns: 3,008; chosen records: 10,000. Source proportions and labels were retained. This benchmark is not validated real-world employee data.

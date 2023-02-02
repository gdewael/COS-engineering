# COS-engineering
Code supporting machine learning for engineering of COS production in *E. coli*.

## Contents
- `data.csv` contains all supporting data, including OD$_{600}$, growth rate $\mu$ and to which split each sample belongs (either test or val0, val1, val2, val3 for either of the cross-validation folds)
- `utils.py` supporting python helper functions to load in data and train models
- `example_script_PLS.py` and `example_script_PLS.py` example script of usage to reproduce our results.

## Dependencies

The following code was tested on `Python version 3.10` with:
```
pandas==1.4.3
numpy==1.21.5
xgboost==1.5.0
matplotlib
seaborn==0.11.2
scikit-learn==1.1.1
shap==0.39.0
```
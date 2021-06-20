# MOOP_CODE
Implementations of **M**ulti-**O**bjective **O**ptimization **P**roblem.

## MMOE
The directory contains the implementation of [Multi-gate Mixture-of-Experts](http://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-) model in PyTorch with the [census-income dataset from UCI](https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)). 

### data
Please Download the [dataset](https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)) and copy/cup the adult.data and adult.names ans paste them in this sub-directory. Run

```python
python3 data.py
```  
to obtain the marital-status.csv for further usage. 
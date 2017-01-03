# Parallel

### pickling problems

Most parallelization frameworks rely on the built-in pickle module which has limited functionality regarding lambda expressions.
Deap relies heavily on those functionalities and thus most parallelization frameworks do not work well with deap.

Dill can handle everything we need and can be monkey patched to replace pickle.

### scoop

Install

`pip install scoop`

and run your app with

`python -m scoop app.py`

 * will detect a PBS environment.
 * sets up a scheduler in the background
 * has pickling problems

### joblib

### dask.distributed

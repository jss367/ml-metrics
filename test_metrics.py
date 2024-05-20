from metrics import calc_precision
import pandas as pd

# pred = [0,0,0,1]
# truth = [1,1,1,1]

tp = 4
fp = 6

precision = calc_precision(tp, fp)

assert precision == 0.4



# test pandas seies
tp = {'cat': 4, 'dog': 3, 'bike': 2}
tp_df = pd.Series(tp)

fp = {'cat': 2, 'dog': 2, 'bike': 2}
fp_df = pd.Series(fp)


precision = calc_precision(tp_df, fp_df)

assert precision == 0.4
print(precision)

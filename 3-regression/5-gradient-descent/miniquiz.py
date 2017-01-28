import numpy as np

a = np.linspace(1,4,4).reshape(2,2)
b = np.linspace(5,8,4).reshape(2,2)



a_sums_column = np.sum(a,axis=0)
b_sums_column = np.sum(b,axis=0)


a_sums_row = np.sum(a,axis=1)
b_sums_row = np.sum(b,axis=1)


a_mean_row = np.mean(a,axis=1)
b_mean_row = np.mean(b,axis=1)

a_mean_column = np.mean(a,axis=0)
b_mean_column = np.mean(b,axis=0)


c = np.ones(2)

print a + c
print b + c
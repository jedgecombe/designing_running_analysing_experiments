import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats import multitest


df = pd.read_csv('data/deviceprefs.csv')

grp = df.groupby(['Pref']).size()

plt.bar(grp.index, grp.values)
plt.title('Single variant')
plt.show()
# chi squared
chi, p = scipy.stats.chisquare(grp.values)

wo_dis = df[df['Disability'] == 1]
wo_dis_grp = wo_dis.groupby('Pref').size()
# binomial
p = scipy.stats.binom_test(wo_dis_grp.values, p=0.5)



# two samples, two response categories
grp_ab_dis = df.groupby(['Pref', 'Disability']).size()
# chi squared
grp_tab = grp_ab_dis.unstack()  # first create a contingency table
chi2, p, dof, _ = scipy.stats.chi2_contingency(grp_tab)


oddsratio, p = scipy.stats.fisher_exact(grp_tab)

import logging

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
from statsmodels.stats import multitest

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s.%(msecs)03d - %(name)s:%(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)


def descr_p(test_descr, p):
    if p < 0.05:
        descr = 'SIGNIFICANT difference'
    else:
        descr = 'no significant difference'
    logger.info(f'\n {test_descr} returns p value of: {p} \n     => {descr}')

# single response category, single sample
prefs_ab = pd.read_csv('data/prefsAB.csv')
grp_ab = prefs_ab.groupby(['Pref']).size()
plt.bar(grp_ab.index, grp_ab.values)
plt.title('Single variant')
plt.show()
# chi squared
chi, p = scipy.stats.chisquare(grp_ab.values)
descr_p('chi squared test', p)
# binomial
p = scipy.stats.binom_test(grp_ab.values, p=0.5)
descr_p('binomial test', p)


# two response categories
prefs_abc = pd.read_csv('data/prefsABC.csv')
grp_abc = prefs_abc.groupby(['Pref']).size()
plt.bar(grp_abc.index, grp_abc.values)
plt.title('Two variants')
plt.show()
# chi squared
chi, p = scipy.stats.chisquare(grp_abc.values)
descr_p('chi squared test - ABC variant', p)
# multinomial
p = scipy.stats.multinomial.pmf(grp_abc.values, len(prefs_abc), p=[1/3]*3)
descr_p('multinomial test', p)
# post hoc pairwise tests
aa = scipy.stats.binom_test(grp_abc.loc['A'], n=len(prefs_abc), p=1/3)
bb = scipy.stats.binom_test(grp_abc.loc['B'], n=len(prefs_abc), p=1/3)
cc = scipy.stats.binom_test(grp_abc.loc['C'], n=len(prefs_abc), p=1/3)
rej, p_vals, _, _ = multitest.multipletests([aa, bb, cc], method='holm')
for num, pv in enumerate(p_vals):
    descr_p(f'post hoc binom {num}', pv)

# NOTE did not find off the shelf version of G test in python

# two samples, two response categories
prefs_ab_sex = pd.read_csv('data/prefsABsex.csv')
grp_ab_sex = prefs_ab_sex.groupby(['Pref', 'Sex']).size()
males = grp_ab_sex.xs('M', level='Sex')
females = grp_ab_sex.xs('F', level='Sex')
plt.bar(males.index, males.values)
plt.title('AB grouping - males')
plt.show()
plt.bar(females.index, females.values)
plt.title('AB grouping - females')
plt.show()
# chi squared
grp_ab_sex_tab = grp_ab_sex.unstack()  # first create a contingency table
chi2, p, dof, _ = scipy.stats.chi2_contingency(grp_ab_sex_tab)
descr_p('chi squared test - AB variant, two sample', p)
# fishers exact test - only works in python with contingency table 2x2
oddsratio, p = scipy.stats.fisher_exact(grp_ab_sex_tab)
descr_p('fishers exact test - AB variant, two sample', p)


# three response categories, two samples
prefs_abc_sex = pd.read_csv('data/prefsABCsex.csv')
grp_abc_sex = prefs_abc_sex.groupby(['Pref', 'Sex']).size()
males_abc = grp_abc_sex.xs('M', level='Sex')
females_abc = grp_abc_sex.xs('F', level='Sex')
plt.bar(males_abc.index, males_abc.values)
plt.title('ABC grouping - males')
plt.show()
plt.bar(females_abc.index, females_abc.values)
plt.title('ABC grouping - females')
plt.show()
# chi squared
grp_abc_sex_tab = grp_abc_sex.unstack()  # first create a contingency table
chi2, p, dof, _ = scipy.stats.chi2_contingency(grp_abc_sex_tab)
descr_p('chi squared test - ABC variant, two sample', p)
# post hoc pairwise tests (just looking at males)
aa = scipy.stats.binom_test(males_abc.loc['A'], n=males_abc.sum(), p=1/3)
bb = scipy.stats.binom_test(males_abc.loc['B'], n=males_abc.sum(), p=1/3)
cc = scipy.stats.binom_test(males_abc.loc['C'], n=males_abc.sum(), p=1/3)
rej, p_vals, _, _ = multitest.multipletests([aa, bb, cc], method='holm')
for num, pv in enumerate(p_vals):
    descr_p(f'post hoc binom, males, variable: {num}', pv)



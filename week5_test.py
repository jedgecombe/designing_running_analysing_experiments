import logging

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison, tukeyhsd
from statsmodels.stats import multitest


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s.%(msecs)03d - %(name)s:%(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

alpha = pd.read_csv('data/alphabets.csv')
alpha['Subject'] = alpha['Subject'].astype('category')
alpha['Alphabet'] = alpha['Alphabet'].astype('category')
logger.info(f'number of alphabets: {len(np.unique(alpha["Alphabet"]))}')

edg = alpha[alpha['Alphabet'] == 'EdgeWrite']
uni = alpha[alpha['Alphabet'] == 'Unistrokes']
graf = alpha[alpha['Alphabet'] == 'Graffiti']

edg_speed = round(edg['WPM'].mean(), 2)
logger.info(f'average WPM of EdgeWrite: {edg_speed}')


def check_p(descr, assumption, p_val):
    if p_val < 0.05:
        logger.info(f'{descr} shows assumption of {assumption} has been violated (p={round(p_val, 4)})')
    else:
        logger.info(f'{descr} shows assumption of {assumption} has NOT been violated (p={round(p_val, 4)})')


# shapiro wilk test of normality on response (although less important than residuals)
for alp in np.unique(alpha['Alphabet'].unique()):
    tmp = alpha[alpha['Alphabet'] == alp]
    w, p = stats.shapiro(tmp['WPM'])
    check_p(descr=f'{alp} response var', assumption='normality', p_val=p)

# above shows NO deviation from normality but more important to check residuals as follows
# fit ANOVA model - check normality of residuals
alpha_lm = ols('WPM ~ C(Alphabet)', data=alpha).fit()
residuals = alpha_lm.resid
w_r, p_r = stats.shapiro(residuals)
check_p(descr='Residuals', assumption='normality', p_val=p_r)
stats.probplot(residuals, dist='norm', plot=plt)  # shows deviation
# plt.show()


# brown-forsythe test
w, p_bf = stats.levene(edg['WPM'], graf['WPM'], uni['WPM'], center='median')
check_p('brown forsythe test', assumption='homogeneity of variance', p_val=p_bf)
# non-significance shows we don't have a violation


# now that we know our assumptions have not been violated, we can fit the ANOVA. This is the omnibus test
alpha_lm = ols('WPM ~ C(Alphabet)', data=alpha).fit()
logger.info(f'ANOVA summary: \n\n {alpha_lm.summary()}')
# Prob (F-statistic) shows that there is some difference between the different Alphabets but does not tell us where the
# difference is. For that we do the pairwise comparisons

# tukey comparison followed by holm adjustment (not sure how to combine the two
mc = MultiComparison(alpha['WPM'], alpha['Alphabet'])
logger.info(f'tukey comparison2: \n {mc.tukeyhsd()}')
comp = mc.allpairtest(stats.ttest_ind, method='Holm')
logger.info(f'holm corrected version: \n {comp[0]}')


# non parametric version of one-way ANOVA
chi, p = stats.kruskal(edg['WPM'], graf['WPM'], uni['WPM'])
check_p(descr='Kruskal chi squared test', assumption='', p_val=p)

# mann whitney
mw, p_eg = stats.mannwhitneyu(edg['WPM'], graf['WPM'], alternative='two-sided')
logger.info(f'mann-whitney stat edg vs. graf: {mw}, p value: {p_eg}')
mw, p_ug = stats.mannwhitneyu(uni['WPM'], graf['WPM'], alternative='two-sided')
logger.info(f'mann-whitney stat uni vs. graf: {mw}, p value: {p_ug}')
mw, p_ue = stats.mannwhitneyu(uni['WPM'], edg['WPM'], alternative='two-sided')
logger.info(f'mann-whitney stat EC vs. PC: {mw}, p value: {p_ue}')
rej, p_vals, _, _ = multitest.multipletests([p_eg, p_ug, p_ue], method='holm')
for num, pv in enumerate(p_vals):
    logger.info(f'post hoc mann-whitney (multi comp adjusted) p value: {round(pv, 4)}, test num: {num}')


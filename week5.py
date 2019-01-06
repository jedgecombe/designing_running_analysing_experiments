import logging

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats import multitest



logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s.%(msecs)03d - %(name)s:%(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

ide = pd.read_csv('data/ide3.csv')
ide['Subject'] = ide['Subject'].astype('category')
ide_counts = ide.groupby(['IDE']).size()
logger.info(f'\n ide dataframe: \n\n{ide.describe().transpose()} \n\n ide counts: \n {ide_counts}')


def descr_dataset(input_dataframe, column, x_ticks, y_ticks):
    # summary stats associated with each ide individually and create histograms
    fig, axarr = plt.subplots(nrows=3)

    for num, idee in enumerate(input_dataframe['IDE'].unique()):
        df = input_dataframe[input_dataframe['IDE'] == idee]
        logger.info(f'summary statistics for site {idee} \n {df.describe().transpose()}')
        sns.distplot(df[column], ax=axarr[num], kde=False, bins=x_ticks)
        axarr[num].set_title(f'IDE {idee} - {column}')
        if x_ticks is not None:
            axarr[num].set_xticks(x_ticks)
            axarr[num].set_yticks(y_ticks)
    plt.show()
    sns.boxplot(x='IDE', y=column, data=ide, palette='Set3')
    plt.show()


descr_dataset(ide, 'Time', x_ticks=np.arange(100, 1001, 100), y_ticks=np.arange(0, 13, 2))
vs_time = ide[ide['IDE'] == 'VStudio']['Time']
ec_time = ide[ide['IDE'] == 'Eclipse']['Time']
pc_time = ide[ide['IDE'] == 'PyCharm']['Time']


def check_p(descr, assumption, p_val):
    if p_val < 0.05:
        logger.info(f'{descr} shows assumption of {assumption} has been violated (p={round(p_val, 4)})')
    else:
        logger.info(f'{descr} shows assumption of {assumption} has NOT been violated (p={round(p_val, 4)})')


# shapiro wilk test of normality on response (although less important than residuals)
w_c, p_c = stats.shapiro(pc_time)
check_p(descr='PyCharm response var', assumption='normality', p_val=p_c)

# above shows deviation from normality but more important to check residuals as follows
# fit ANOVA model - check normality of residuals
ide_lm = ols('Time ~ C(IDE)', data=ide).fit()
residuals = ide_lm.resid
w_r, p_r = stats.shapiro(residuals)
check_p(descr='Residuals', assumption='normality', p_val=p_r)
stats.probplot(residuals, dist='norm', plot=plt)  # shows deviation
plt.show()

# after concluding non-normality of residuals, check for log normality
# use Kolmogorv Smirnov test for this
# note that results are slightly different than used in vid, think this is to do with parameters passed
ks_c, p_c = stats.kstest(pc_time, 'lognorm', stats.lognorm.fit(pc_time, floc=0),
                         alternative='two-sided', mode='asymp')
check_p(descr='PyCharm check aganst log-normality', assumption='log-normality', p_val=p_c)

# seem to be lognormal, so create log time column (which will satisfy assumptions)
ide['log_time'] = np.log(ide['Time'])
vs_log_time = ide[ide['IDE'] == 'VStudio']['log_time']
ec_log_time = ide[ide['IDE'] == 'Eclipse']['log_time']
pc_log_time = ide[ide['IDE'] == 'PyCharm']['log_time']
w_c, p_c = stats.shapiro(pc_log_time)
check_p(descr='PyCharm response var (log-transformed)', assumption='normality', p_val=p_c)
# this passes - now check residuals
ide_lm_log = ols('log_time ~ C(IDE)', data=ide).fit()
residuals_log = ide_lm_log.resid
w_r, p_r = stats.shapiro(residuals_log)
check_p(descr='Residuals (from log-transformed)', assumption='normality', p_val=p_r)
stats.probplot(residuals_log, dist='norm', plot=plt)
plt.show()

# brown-forsythe te4st
w, p_bf = stats.levene(vs_log_time, ec_log_time, pc_log_time, center='median')
check_p('brown forsythe test', assumption='homogeneity of variance', p_val=p_bf)
# non-significance shows we don't have a violation

# now that we know our assumptions have not been violated, we can fit the ANOVA. This is the omnibus test
ide_lm = ols('log_time ~ C(IDE)', data=ide).fit()
logger.info(f'ANOVA summary: \n\n {ide_lm.summary()}')
# Prob (F-statistic) shows that there is some difference between the different IDEs but does not tell us where the
# difference is. For that we do the pairwise comparisons


tukey = pairwise_tukeyhsd(endog=ide['log_time'],     # response
                          groups=ide['IDE'],   # group array
                          alpha=0.05)
tukey.plot_simultaneous()    # Plot group confidence intervals
logger.info(f'tukey comparison: \n {tukey.summary()}')   # See test summary

# non parametric version of one-way ANOVA
chi, p = stats.kruskal(vs_log_time, ec_log_time, pc_log_time)
check_p(descr='Kruskal chi squared test (log-transformed)', assumption='', p_val=p)
# this would be the same in log or non-log as it is a rank based test

# mann whitney
mw, p_ve = stats.mannwhitneyu(vs_time, ec_time, alternative='two-sided')
logger.info(f'mann-whitney stat VS vs. EC: {mw}, p value: {p_ve}')
mw, p_vp = stats.mannwhitneyu(vs_time, pc_time, alternative='two-sided')
logger.info(f'mann-whitney stat VS vs. PC: {mw}, p value: {p_vp}')
mw, p_ep = stats.mannwhitneyu(ec_time, pc_time, alternative='two-sided')
logger.info(f'mann-whitney stat EC vs. PC: {mw}, p value: {p_ep}')
rej, p_vals, _, _ = multitest.multipletests([p_ve, p_vp, p_ep], method='holm')
for num, pv in enumerate(p_vals):
    logger.info(f'post hoc mann-whitney (multi comp adjusted) p value: {pv}, test num: {num}')

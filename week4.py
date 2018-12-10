import logging

import matplotlib.pyplot as plt
import pandas as pd
# import pylab
import numpy as np
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s.%(msecs)03d - %(name)s:%(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

ide = pd.read_csv('data/ide2.csv')
ide['Subject'] = ide['Subject'].astype('category')
ide_counts = ide.groupby(['IDE']).size()
logger.info(f'\n ide dataframe: \n\n{ide.describe().transpose()} \n\n ide counts: \n {ide_counts}')


def descr_dataset(input_dataframe, column, x_ticks, y_ticks):
    # summary stats associated with each ide individually and create histograms
    fig, axarr = plt.subplots(nrows=2)

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

# t test
t, p = stats.ttest_ind(vs_time, ec_time, equal_var=True)
logger.info(f't stat: {t}, p value: {p}')


def check_p(descr, assumption, p_val):
    if p_val < 0.05:
        logger.info(f'{descr} shows assumption of {assumption} has been violated (p={p_val})')
    else:
        logger.info(f'{descr} shows assumption of {assumption} has NOT been violated (p={p_val})')


# shapiro wilk test of normality on response (although less important than residuals)
w_v, p_v = stats.shapiro(vs_time)
w_e, p_e = stats.shapiro(ec_time)
check_p(descr='VStudio response var', assumption='normality', p_val=p_v)
check_p(descr='Eclipse response var', assumption='normality', p_val=p_e)

# fit ANOVA model - check normality of residuals
ide_lm = ols('Time ~ C(IDE)', data=ide).fit()
residuals = ide_lm.resid
w_r, p_r = stats.shapiro(residuals)
check_p(descr='Residuals', assumption='normality', p_val=p_r)
stats.probplot(residuals, dist='norm', plot=plt)  # shows deviation
# plt.show()


# after concluding non-normality of residuals, check for log normality
# use Kolmogorv Smirnov test for this
# note that results are slightly different than used in vid, think this is to do with parameters passed
ks_v, p_v = stats.kstest(vs_time, 'lognorm', stats.lognorm.fit(vs_time, floc=0),
                         alternative='two-sided', mode='asymp')
ks_e, p_e = stats.kstest(ec_time, 'lognorm', stats.lognorm.fit(ec_time, floc=0),
                         alternative='two-sided', mode='asymp')
check_p(descr='VStudio check aganst log-normality', assumption='log-normality', p_val=p_v)
check_p(descr='Eclipse check aganst log-normality', assumption='log-normality', p_val=p_e)


# test for homogeneity of variance
# levene's test
w, p_lv = stats.levene(vs_time, ec_time, center='mean')
# brown-forsythe te4st
w, p_bf = stats.levene(vs_time, ec_time, center='median')
check_p('levene test', assumption='homogeneity of variance', p_val=p_lv)
check_p('brown forsythe test', assumption='homogeneity of variance', p_val=p_bf)
# t test
t, p = stats.ttest_ind(vs_time, ec_time, equal_var=False)
logger.info(f't stat (welchs): {t}, p value: {p}')


# to deal with data being shown to not be normal above (but does not violate log-normal condition) transform
ide['log_time'] = np.log(ide['Time'])
# describe, as done with above but with log
descr_dataset(ide, 'log_time', x_ticks=np.arange(5, 7.5, 0.25), y_ticks=np.arange(0, 8, 1))

# re-test for normality based on transformed data
vs_log_time = ide[ide['IDE'] == 'VStudio']['log_time']
ec_log_time = ide[ide['IDE'] == 'Eclipse']['log_time']
w_v, p_v = stats.shapiro(vs_log_time)
w_e, p_e = stats.shapiro(ec_log_time)
check_p(descr='VStudio response var (log transformed)', assumption='normality', p_val=p_v)
check_p(descr='Eclipse response var (log transformed)', assumption='normality', p_val=p_e)
# fit ANOVA model - check normality of residuals
ide_lm_log = ols('log_time ~ C(IDE)', data=ide).fit()
residuals_log = ide_lm_log.resid
w_r, p_r = stats.shapiro(residuals_log)
check_p(descr='Residuals (from log transformed response)', assumption='normality', p_val=p_r)
stats.probplot(residuals_log, dist='norm', plot=plt)  # shows deviation
plt.show()
# re test on homoscedasticity to check this has not now been violated
w, p_bf = stats.levene(vs_log_time, ec_log_time, center='median')
check_p('brown forsythe test (log transformed response)', assumption='homogeneity of variance', p_val=p_bf)
# assumptions are not violated on transformed data so can issue t test with equal variance assumptions
t, p = stats.ttest_ind(vs_log_time, ec_log_time, equal_var=True)
logger.info(f't stat (log transformed data): {t}, p value: {p}')


# mann whitney
mw, p = stats.mannwhitneyu(vs_time, ec_time, alternative='two-sided')
logger.info(f'mann-whitney stat (non-transformed data): {mw}, p value: {p}')
mw, p = stats.mannwhitneyu(vs_log_time, ec_log_time, alternative='two-sided')
logger.info(f'mann-whitney stat (log-transformed data): {mw}, p value: {p}')
# the above are the same as taking the log doesn't fundamentally change the rank of the data

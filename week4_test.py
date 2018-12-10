import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import lognorm
from scipy.stats import kstest
from scipy.stats import ttest_ind
from scipy.stats import ranksums
from scipy.stats import mannwhitneyu
from statsmodels.formula.api import ols



path = 'data/'
df = pd.read_csv(path+'designtime.csv')

print(len(df))

sns.boxplot(x='Tool', y='Time', data=df, palette='Set3')
plt.show()



# Shapiro-Wilk test
# -----------------

# Shapiro-Wilk test for Illustrator
stat, p = shapiro(x=df[df['Tool']=='Illustrator']['Time'])
print('Shapiro-Wilk test for Illustrator: stat={:.4f}, p_value={:.4f}'.format(stat, p))


# Shapiro-Wilk on resids
Time = df['Time']
Tool = df['Tool']=='Illustrator'

formula = 'Time ~ C(Tool)'

lm = ols(formula, df).fit()

stat, p = shapiro(lm.resid)
print('Shapiro-Wilk test for residuals: stat={:.4f}, p_value={:.4f}'.format(stat, p))



# Brown-Forsythe / Levene test
# ----------------------------

stat, p = levene(df[df['Tool']=='Illustrator']['Time'], df[df['Tool']=='InDesign']['Time'], center='median')
print('Levene test of homoscedasticity: stat={:.4f}, p_value={:.4f}'.format(stat, p))




# Kolmogorov-Smirnov test
# -----------------------

# Kolmogorov-Smirnov goodness-of-fit test on the lognormal transformation
# Note, the following code produces same statistic value as R's test, but different p-value.
# The kstest function in SciPy does not have an option to compute the exact p-value.  https://stackoverflow.com/a/53178125/5056689
# we can add a mode='asymp' arg to kstest, but given our sample size is small it will still not be accueate. TO DO: research further.

illustrator = df[df['Tool']=='Illustrator']['Time']
sigma, loc, scale = lognorm.fit(illustrator, floc=0)

stat, p = kstest(illustrator, 'lognorm', args=(sigma, 0, scale), alternative='two-sided')
print('KS test: stat={:.4f}, p_value={:.4f}'.format(stat, p))




# Welch's t-test
# --------------
# Iindependent-samples t-test on the log-transformed Time response.
# Use the Welch version for unequal variances.

df['Time_lognormal'] = df['Time'].apply(np.log)
in_design = df[df['Tool'] == 'InDesign']
print(f'mean of log transformed time: {in_design["Time_lognormal"].mean()}')

stat, p = ttest_ind(a=df[df['Tool']=='Illustrator']['Time_lognormal'], b=df[df['Tool']=='InDesign']['Time_lognormal'], equal_var=False)
print('Welch t-test: stat={:.4f}, p_value={:.4f}'.format(stat, p))




# Nonparametric Mann-Whitney U test
# Wilcoxon rank-sum test
# ---------------------------------
# In this case we need the Wilcoxon rank-sum statistic for two samples.

indesign = df[df['Tool']=='InDesign']['Time']

stat, p = ranksums(illustrator, indesign)
print('Wilcoxon rank-sum test: stat={:.4f}, p_value={:.4f}'.format(stat, p))

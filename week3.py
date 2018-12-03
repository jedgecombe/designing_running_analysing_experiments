import logging

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns



logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s.%(msecs)03d - %(name)s:%(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

pgviews = pd.read_csv('data/pgviews.csv')

# first look at the data
pgviews['Subject'] = pgviews['Subject'].astype('category')
site_counts = pgviews.groupby(['Site']).size()
logger.info(f'\n pgviews dataframe: \n\n{pgviews.describe().transpose()} \n\n site views: \n {site_counts}')

# summary stats associated with each site individually and create histograms
fig, axarr = plt.subplots(nrows=2)
for num, site in enumerate(pgviews['Site'].unique()):
    site_df = pgviews[pgviews['Site'] == site]
    logger.info(f'summary statistics for site {site} \n {site_df.describe().transpose()}')
    sns.distplot(site_df['Pages'], ax=axarr[num], kde=False)
    axarr[num].set_title(f'Site {site} - page views')
plt.show()

# boxplot
sns.boxplot(x='Site', y='Pages', data=pgviews, palette='Set3')
plt.show()

# t test
t, p = stats.ttest_ind(pgviews[pgviews['Site'] == 'A']['Pages'], pgviews[pgviews['Site'] == 'B']['Pages'],
                       equal_var=True)


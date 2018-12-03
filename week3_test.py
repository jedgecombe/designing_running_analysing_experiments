import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s.%(msecs)03d - %(name)s:%(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

df = pd.read_csv('data/timeonsite.csv')
logger.info(f'number of subjects: {len(df)}')

subjects = df.groupby(['Subject']).size()
two_sites = subjects[subjects > 1]
logger.info(f'number exposed to both: {len(two_sites)}')

web_b = df[df['Site'] == 'B']
med_b = np.median(web_b['Time'])
logger.info(f'web B median time: {med_b}')

web_a = df[df['Site'] == 'A']
std_a = np.std(web_a['Time'])
logger.info(f'web A STD: {round(std_a, 2)}')

t, p = stats.ttest_ind(web_a['Time'], web_b['Time'], equal_var=True)
logger.info(f't stat: {round(t, 2)}, p value: {round(p, 4)}')

dof = logger.info(f'degrees of freedom: {len(df) - 2}')
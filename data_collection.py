# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 14:43:10 2020

@author: ayushjain
"""

import glassdoor_scraper as gs
import pandas as pd

path = "C:/Users/ayushjain9/Documents/Projects/ds_project/chromedriver.exe"

df = gs.get_jobs('data scientist',5, False, path, 15) # for 5 records

df.csv('glassdoor_job.csv', index = False)
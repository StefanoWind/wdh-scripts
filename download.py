# -*- coding: utf-8 -*-
"""
General data downloader
"""

import os
cd=os.getcwd()
import sys
sys.path.append(os.path.join(cd,'dap-py'))
from doe_dap_dl import DAP
from matplotlib import pyplot as plt
import yaml
import matplotlib
import warnings
import pandas as pd

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12

plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs
source_config=os.path.join(cd,'config.yaml')

if len(sys.argv)==1:
    source_order=os.path.join(cd,'data','download_order2.xlsx')#source of download order 
    save_path=os.path.join('D:/experimental-dataset')#where to save data
else:
    source_order=sys.argv[1]
    save_path=sys.argv[2]

#%% Initialization
with open(os.path.join(cd,'config.yaml'), 'r') as fid:
    config = yaml.safe_load(fid)
    
DO=pd.read_excel(source_order)
a2e = DAP('wdh.energy.gov',confirm_downloads=False)

#%% Main
for i in DO.index:
    try:
        channel=DO['channel'][i]
        sdate=str(DO['sdate'][i])
        edate=str(DO['edate'][i])
        ext=str(DO['ext'][i])
        ftype=DO['ftype'][i]
        MFA=DO['MFA'][i]
        
        if MFA==False:#if multi factor authentication is needed
            a2e.setup_cert_auth(username=config['username'], password=config['password'])
        else:
            a2e.setup_two_factor_auth(username=config['username'], password=config['password'])
            
        if ext=='nan':
            _filter = {
                'Dataset': channel,
                'date_time': {
                    'between': [sdate,edate]
                },
                'file_type':ftype,
            }
        else:
            _filter = {
                'Dataset': channel,
                'date_time': {
                    'between': [sdate,edate]
                },
                'file_type':ftype,
                'ext1': ext
            }
         
        os.makedirs(os.path.join(save_path,channel),exist_ok=True)
        a2e.download_with_order(_filter, path=os.path.join(save_path,channel),replace=False)
    except:
        print(f'Request {i} in {source_order} failed.')
   
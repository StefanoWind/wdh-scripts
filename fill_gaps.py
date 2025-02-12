# -*- coding: utf-8 -*-
"""
Download raw hpl files if a0 file is not present
"""

import os
cd=os.getcwd()
import sys
sys.path.append(os.path.join(cd,'dap-py'))
from doe_dap_dl import DAP
from matplotlib import pyplot as plt
import yaml
import matplotlib
import glob
import re
import numpy as np
import utils as utl
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
    source1=os.path.join(cd,'data/crosswind/nwtc.lidar.z01.a0/*nc')#source of level 1 files 
    channel1='nwtc.lidar.z01.a0'
    channel2='crosswind/nwtc.lidar.z01.00'
    time_tol=60#[s]
    MFA=False
    ftype='hpl'
    ext='stare'
    save_path=os.path.join(cd,'data')
else:
    source_order=sys.argv[1]
    save_path=sys.argv[2]

#%% Initialization
with open(os.path.join(cd,'config.yaml'), 'r') as fid:
    config = yaml.safe_load(fid)

a2e = DAP('a2e.energy.gov',confirm_downloads=False)
if MFA==False:#if multi factor authentication is needed
    a2e.setup_cert_auth(username=config['username'], password=config['password'])
else:
    a2e.setup_two_factor_auth(username=config['username'], password=config['password'])

#%% Main
files1=sorted(glob.glob(source1))
time1=np.array([],dtype='datetime64')
for f in files1:
    match = re.search(r"\.(\d{8})\.(\d{6})\.", f)
    date_str, time_str = match.groups()
    time1=np.append(time1,np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}T{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"))

sdate=str(np.min(time1)).replace('-','').replace('T','').replace(':','')
edate=str(np.max(time1)).replace('-','').replace('T','').replace(':','')
files2=utl.dap_search(a2e,channel2,sdate,edate,ftype,ext)#search files

files_download=[]
for f in files2:
    date_str=f['data_date']
    time_str=f['data_time']
    t2=np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}T{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}")
    if np.min(np.abs(time1-t2))>np.timedelta64(time_tol, 's'):
        files_download.append(f)
        
a2e.download_files(files_download,os.path.join(save_path,channel2),replace=False)

   
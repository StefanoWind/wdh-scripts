# -*- coding: utf-8 -*-
"""
Download and clamp netcdf datamfrom multiple channels


"""
import os
cd=os.path.dirname(__file__)
import sys
import glob
import yaml
import xarray as xr
import re
from datetime import datetime
import numpy as np


#%% Inputs

#dataset
channels=['wfip3/barg.assist.tropoe.z01.c0',
          'wfip3/bloc.assist.tropoe.z01.c0',
          'wfip3/nant.assist.tropoe.z01.c0',
          'wfip3/rhod.assist.tropoe.z01.c0']

variables=['temperature','sigma_temperature','waterVapor','sigma_waterVapor','gamma','rmsr','lwp','cbh']

MFA=False#use multi-factor authentication (for CRADA-protected channels)

sdate='20240501000000'#start date
edate='20240801000000'#end date

#%% Initialization

#config
with open(os.path.join(cd,'config.yaml'), 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(os.path.join(cd,'utils'))
sys.path.append(os.path.join(cd,'dap-py'))
import utils as utl
from doe_dap_dl import DAP
    
#WDH setup
a2e = DAP('a2e.energy.gov',confirm_downloads=False)
if MFA==False:
    a2e.setup_cert_auth(username=config['username'], password=config['password'])
else:
    a2e.setup_two_factor_auth(username=config['username'], password=config['password'])

#%% Main
a2e.setup_basic_auth(username=config['username'], password=config['password'])
for channel in channels:
    _filter = {
        'Dataset': channel,
        'date_time': {
            'between': [sdate,edate]
        },
        'file_type':'nc'
    }
    
    utl.mkdir(os.path.join('data',channel))
    # a2e.download_with_order(_filter, path=os.path.join('data',channel),replace=False)
    
#%% Output
for channel in channels:
    
    files= np.array(sorted(glob.glob(os.path.join('data',channel,'*nc'))))
    
    if len(files)>0:
        t_file=[]
        for f in files:
            match = re.search(r'\d{8}\.\d{6}', f)
            t=datetime.strptime(match.group(0),'%Y%m%d.%H%M%S')
            t_file=np.append(t_file,t)
        
        sel=(t_file>=datetime.strptime(sdate,'%Y%m%d%H%M%S'))*(t_file<=datetime.strptime(edate,'%Y%m%d%H%M%S'))
        if np.sum(sel)>0:
            Data_all=None
            for f in files[sel]:
                try:
                    Data= xr.open_dataset(f)[variables]
                    if Data_all is None:
                        Data_all = Data # First file initializes the dataset
                    else:
                        try:
                            Data_all = xr.concat([Data_all, Data], dim="time")  # Concatenate along time
                        except:
                            print('Could not concatenate '+f)
                except:
                    print('Could not load '+f)
            if Data_all is not None:
                Data_all.to_netcdf(os.path.join(cd,'data/'+channel.split('/')[1]+'.'+sdate[:8]+'.'+sdate[8:]+'.'+edate[:8]+'.'+edate[8:]+'.nc'))
        else:
            print('No files found in '+channel+' in the specified period')
    else:
        print('No files found in '+channel+' in the specified period')
        
    

# -*- coding: utf-8 -*-
"""
Search, download data and plot timelines from Wind Data hub
"""

import os
cd=os.getcwd()
import sys
sys.path.append(os.path.join(cd,'dap-py'))
from doe_dap_dl import DAP
from datetime import datetime
from datetime import timedelta
import numpy as np
from matplotlib import pyplot as plt
import yaml
import matplotlib
import warnings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12

plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs

#time range
sdate='20240501'#[%Y%m%d] start date
edate='20241114'#[%Y%m%d] end date

#channel names
channels={'barg':'wfip3/barg.assist.tropoe.z01.c0'}

#file formats
formats={'barg':'nc',
          'rhod':'nc'}

#extension e.g. user1 or rhi or assistsummary (put '' if not applicable)
ext={'barg':'',
    'rhod':''}

MFA=False#use multi-factor authentication (for CRADA-protected channels)
download=False#download selected data

#graphics
time_res=3600*24#[s] time duration of one file in the timeline

#%% Functions
def datenum(string,format="%Y-%m-%d %H:%M:%S.%f"):
    '''
    Turns string date into unix timestamp
    '''
    from datetime import datetime
    num=(datetime.strptime(string, format)-datetime(1970, 1, 1)).total_seconds()
    return num

def datestr(num,format="%Y-%m-%d %H:%M:%S.%f"):
    '''
    Turns Unix timestamp into string
    '''
    from datetime import datetime
    string=datetime.utcfromtimestamp(num).strftime(format)
    return string

def dap_search(channel,sdate,edate,file_type,ext1,time_search=30):
    '''
    Wrapper for a2e.search to avoid timeout:
        Inputs: channel name, start date, end date, file format, extention in WDH name, number of days scanned at each loop
        Outputs: list of files mathing the criteria
    '''
    dates_num=np.arange(datenum(sdate,'%Y%m%d%H%M%S'),datenum(edate,'%Y%m%d%H%M%S'),time_search*24*3600)
    dates=[datestr(d,'%Y%m%d%H%M%S') for d in dates_num]+[edate]
    search_all=[]
    for d1,d2 in zip(dates[:-1],dates[1:]):
        
        if ext1!='':
            _filter = {
                'Dataset': channel,
                'date_time': {
                    'between': [d1,d2]
                },
                'file_type': file_type,
                'ext1':ext1, 
            }
        else:
            _filter = {
                'Dataset': channel,
                'date_time': {
                    'between': [d1,d2]
                },
                'file_type': file_type,
            }
        
        search=a2e.search(_filter)
        
        if search is None:
            print('Invalid authentication')
            return None
        else:
            search_all+=search
    
    return search_all

#%% Initialization
with open(os.path.join(cd,'config.yaml'), 'r') as fid:
    config = yaml.safe_load(fid)

#WDH setup
a2e = DAP('a2e.energy.gov',confirm_downloads=False)
if MFA==False:
    a2e.setup_cert_auth(username=config['username'], password=config['password'])
else:
    a2e.setup_two_factor_auth(username=config['username'], password=config['password'])

#%% Main
plt.figure(figsize=(18,len(channels)))
ctr=1
for s in channels:
    
    files=dap_search(channels[s],sdate+'000000',edate+'000000',formats[s],ext[s])#search files

    #download
    if download:
        a2e.download_files(files,os.path.join(cd,'data',channels[s]),replace=False)
        
    #extract info
    dates=[]
    size=0
    for f in files:
        dates=np.append(dates,datetime.strptime(f['date_time'],'%Y%m%d%H%M%S'))
        size+=f['size']/10**9
   
    #plot
    ax=plt.subplot(len(channels),1,ctr)     
    for d in dates:
        ax.fill_between([d,d+timedelta(seconds=time_res)],y1=[1,1],y2=0,color='g')
    plt.title(channels[s]+f': {len(files)} total files, {np.round(size,2)} Gb')
    
    plt.xlim([datetime.strptime(sdate, '%Y%m%d'),datetime.strptime(edate, '%Y%m%d')])
    plt.yticks([])
    ctr+=1
     
plt.tight_layout()
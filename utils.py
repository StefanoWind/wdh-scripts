# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:07:59 2025

@author: sletizia
"""
import numpy as np
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

def dap_search(a2e,channel,sdate,edate,file_type,ext1,time_search=30):
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

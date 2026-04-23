# wdh-scripts

## General
Useful scripts to download data from the Wind Data Hub.

## Set up
- Clone on your local machine.
- Clone the dap-py repository on your local machine from [https://github.com/StefanoWind/dap-py].
- Install the dap-py repository into your Python environment:
  'cd dap-py'
  'pip install .'
- Create the config.yaml file as:
username: 'myusername'  
password: 'mypassword'

## Use
Run the download.py as follows:

`python download.py <absolute_path_to_download_order_file> <absolute_path_where_data_are_saved>`

The download order file is an xlsx with columns:
- channel: channel on the WDH, including project
- sdate: start date
- edate: end date date
- ext: file category
- ftype: file type
- MFA: if multi-factor authentication is needed

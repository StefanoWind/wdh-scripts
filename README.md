# wdh-scripts
Useful scripts for the Wind Data Hub.

Clone on your local machine and then add a configuration file containing your Wind Data Hub credentials. 

The yaml file can be created as a text file like:

username: 'myusername'  
password: 'mypassword'

and then saved as "config.yaml".

For the general download, use the download.py as follows:

`python download.py <absolute_path_to_download_order_file> <absolute_path_where_data_are_saved>`

The download order file is an xlsx with columns:
- channel: channel on the WDH, including project
- sdate: start date
- edate: end date date
- ext: file category
- ftype: file type
- MFA: if multi-factor authentication is needed

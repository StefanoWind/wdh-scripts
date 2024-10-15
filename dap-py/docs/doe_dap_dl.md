# doe_dap_dl Package

This package contains the DAP module and wraps all other modules to make importing easy. Information on the plotting module is in plot/README.md. For more information on the DAP module, keep reading.

## DAP Module

The DAP module is a high-level interface that allows the programmer to use our API to authenticate, search, and download the data they want in very few lines of code.

## Installation

The doe_dap_dl package can be installed via pip: `pip install doe-dap-dl`

## Setup

The following examples demonstrate how to download data via A2e.

First, import the main module:

```python
from doe_dap_dl import DAP
```

Then, create an instance of the `DAP` class. The constructor takes one required argument: the hostname of the service from which you want to download data.

```python
a2e = DAP('a2e.energy.gov')
```

And that's it! Setup is complete. All future methods will revolve around this `a2e` object. The constructor also accepts the following optional arguments

- `cert_path` (str): path to authentication certificate file.
- `save_cert_dir` (str): Path to directory where certificates are stored.
- `download_dir` (str): Path to directory where files will be downloaded.
- `quiet` (bool): suppresses output print statemens. Useful for scripting. Defaults to False.
- `spp` (bool): If this is a dap for the Solid Phase Processing data. Defaults to False.
- `confirm_downloads` (bool): Whether or not to confirm before downloading. Defaults to True.

## Authentication

Authentication is simple. This module supports both __basic__ and __certificate__ authentication protocols. The basic method does not use a certificate, expires more quickly, and does not support two-factor authentication. __Proper authentication is required to use this module.__

### Certificates

The default name for an authetication certificate is `.<host name>.cert`, for example `.a2e.energy.gov.cert`. The following is the precedence of the different ways to provide a certificate's location.

1. `cert_path`
2. If `save_cert_dir` was provided, `DAP` will look for a certificate at `<save_cert_dir>/.<host name>.cert`
3. If the environment variable `DAP_CERT_DIR` exists, `DAP` will look for a certificate at `$DAP_CERT_DIR/.<host name>.cert`
4. Otherwise, `DAP` will look for a certificate at `~/doe_dap_dl/certs/.<host name>.cert`

If the certificate is valid, the module will renew it. If you don't have a valid certificate, you will have to authenticate via one of the following methods:

#### `a2e.setup_basic_auth(username=None, password=None)`

Creates authentication with a username and password. The arguments are optional, but the module will prompt for them if omitted.

#### `a2e.setup_cert_auth(username=None, password=None)`

Creates authentication with a certificate that can be used for future authentication. The certificate is stored in a file named `.<host name>.cert` (e.g. `.a2e.energy.gov.cert`). Returns whether or not a valid certificate was created.

#### `a2e.setup_two_factor_auth(username=None, password=None, authcode=None)`

Creates a certificate, but with two-factor authentication. The authcode is the 6-digit password code from Google Authenticator. This is the highest authentication level available, and is necessary to search for and download certain datasets. The certificate is stored in a file named `.<host name>.cert`. Returns whether or not a valid certificate was created.

### Searching for Files

To search for files, one must first construct a filter. Below is an example filter.

```python
filter = {
    'Dataset': 'wfip2/lidar.z01.b0',
    'date_time': {
        'between': ['20160101000000', '20160104000000']
    },
    'file_type': 'nc'
}
```

The documentation for constructing the filter argument can be found in `docs/download-README.md`

Now simply call this function:

#### `a2e.search(filter_arg, table='inventory', latest=True)`

The `'inventory'` option returns a list of files that match the filter. Filters that return large lists of files may time out, or return an empty list (despite the query matching many files). To avoid this, you can request an accounting of files by calling the function with `table='stats'`.

By default, only the latest files are considered for the search. If you'd like to include older files, you can use `latest=False`. Old files may not be downloadable.

### Downloading Files

There are three functions you can use to download files using this module.

#### Download with a list of files

An inventory search returns a list of files. These can be provided to the following function:

#### `a2e.download_files(files, path='/var/tmp/', replace=False)`

The path specifies where the module will download files. The replace flag determines whether the module should replace files that already exist in the download directory. By default, the module will not replace existing files.

##### Example

```python
filter = {
    'Dataset': 'wfip2/lidar.z04.a0',
    'date_time': {
        'between': ['20151001000000', '20151004000000']
    },
    'file_type': 'nc'
}

file_names = a2e.search(filter, table='Inventory')
files = a2e.download_files(file_names)
```

All the download functions return a list of paths to the downloaded files.

#### Download files directly from a search

Inventory searches fail with large numbers of files. This method will avoid creating a list of files and instead download using a search query. The module will prompt you to confirm that you want to download the files, although it won't say how much space the files will take up, so caution is recommended.

The DAP function is:

#### `a2e.download_search(filter_arg, path='/var/tmp/', force=False)`

##### Example

```python
filter = {
    'Dataset': 'wfip2/lidar.z04.a0',
    'date_time': {
        'between': ['20151001000000', '20151004000000']
    },
    'file_type': 'nc'
}

files = a2e.download_search(filter)
```

Provided with a [filter argument](https://github.com/a2edap/dap-py/blob/master/a2e/download-README.md), search the Inventory table and download the files in s3. I heard a rumor through the grapevine that only files in s3 will be downloaded, so if you think some data could be someone else, use the next download method.

#### Download by placing an order

Placing an order is required to download files that are not in s3. The following function takes a filter like `download_search()` but places an order before downloading.

#### `a2e.download_with_order(filter_arg, path='/var/tmp/', force=False)`

Like `download_search()`, the code will prompt you to confirm that you want to download the files.

```python
filter = {
    'Dataset': 'wfip2/lidar.z04.a0',
    'date_time': {
        'between': ['20151001000000', '20151004000000']
    },
    'file_type': 'nc'
}

a2e.download_with_order(filter)
```

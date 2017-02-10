import pytest
import os
import tarfile
import xarray as xr
import numpy as np
import dask
from contextlib import contextmanager
import py
import tempfile
from glob import glob
from shutil import copyfile
import dask
import dask.array as dsa

import xmitgcm

_TESTDATA_FILENAME = 'testdata.tar.gz'
_TESTDATA_ITERS = [39600, ]
_TESTDATA_DELTAT = 86400

_EXPECTED_GRID_VARS = ['XC', 'YC', 'XG', 'YG', 'Zl', 'Zu', 'Z', 'Zp1', 'dxC',
                       'rAs', 'rAw', 'Depth', 'rA', 'dxG', 'dyG', 'rAz', 'dyC',
                       'PHrefC', 'drC', 'PHrefF', 'drF',
                       'hFacS', 'hFacC', 'hFacW']


_xc_meta_content = """ simulation = { 'global_oce_latlon' };
 nDims = [   2 ];
 dimList = [
    90,    1,   90,
    40,    1,   40
 ];
 dataprec = [ 'float32' ];
 nrecords = [     1 ];
"""


_experiments = {
    'global_oce_llc90': {'geometry': 'llc',
                         'ref_date': "1948-01-01 12:00:00",
                         'delta_t': 3600,
                         'expected_time':[
                             (0, np.datetime64('1948-01-01T12:00:00.000000000')),
                             (1, np.datetime64('1948-01-01T20:00:00.000000000'))],
                         'shape': (50, 13, 90, 90), 'test_iternum': 8,
                         'dtype': np.dtype('f4'),
                         'expected_values': {'XC': ((2,3,5), -32.5)},
                         'diagnostics': ('state_2d_set1', ['ETAN', 'SIarea',
                            'SIheff', 'SIhsnow', 'DETADT2', 'PHIBOT',
                            'sIceLoad', 'MXLDEPTH', 'oceSPDep', 'SIatmQnt',
                            'SIatmFW', 'oceQnet', 'oceFWflx', 'oceTAUX',
                            'oceTAUY', 'ADVxHEFF', 'ADVyHEFF', 'DFxEHEFF',
                            'DFyEHEFF', 'ADVxSNOW', 'ADVySNOW', 'DFxESNOW',
                            'DFyESNOW', 'SIuice', 'SIvice'])},
}

def setup_mds_dir(tmpdir_factory, request):
    """Helper function for setting up test cases."""
    expt_name = request.param
    expected_results = _experiments[expt_name]
    target_dir = str(tmpdir_factory.mktemp('mdsdata'))
    data_dir = os.path.dirname(request.module.__file__)
    return untar(data_dir, expt_name, target_dir), expected_results


def untar(data_dir, basename, target_dir):
    """Unzip a tar file into the target directory. Return path to unzipped
    directory."""
    datafile = os.path.join(data_dir, basename + '.tar.gz')
    if not os.path.exists(datafile):
        raise IOError('Could not find data file %s' % datafile)
    tar = tarfile.open(datafile)
    tar.extractall(target_dir)
    tar.close()
    # subdirectory where file should have been untarred.
    # assumes the directory is the same name as the tar file itself.
    # e.g. testdata.tar.gz --> testdata/
    fulldir = os.path.join(target_dir, basename)
    if not os.path.exists(fulldir):
        raise IOError('Could not find tar file output dir %s' % fulldir)
    # the actual data lives in a file called testdata
    return fulldir

# find the tar archive in the test directory
# http://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
@pytest.fixture(scope='module', params=_experiments.keys())
def all_mds_datadirs(tmpdir_factory, request):
    return setup_mds_dir(tmpdir_factory, request)


def test_mds_to_netcdf(all_mds_datadirs):
    dirname, expected = all_mds_datadirs
    ds = xmitgcm.open_mdsdataset(dirname, iters='all',
                                 prefix=['Eta', 'U', 'V', 'T', 'S'],
                                 read_grid=True, swap_dims=False,
                                 ref_date=expected['ref_date'],
                                 delta_t=expected['delta_t'],
                                 default_dtype=expected['dtype'],
                                 geometry=expected['geometry'])
    xmitgcm.mds_to_netcdf(ds, ['Eta', 'U', 'V', 'T', 'S'],
	                      lat_min=-60, lat_max=0, lon_min=150, lon_max=180,
                          output_dir=dirname, extract_grid=True,
                          concatenate='daily', overwrite=True)
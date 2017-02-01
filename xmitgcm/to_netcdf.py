"""
Useful functions to extract the mds binaries into netcdf files using
from xarray.Dataset
"""
# Python 3 compatiblity
from __future__ import print_function, division
# Xarray
import xarray as xr
# Internals
import os


_DROPPED_COORDS = ['dxC', 'dyC', 'dxG', 'dyG',
                   'rA', 'rAs', 'rAw', 'rAz',
                   'k_p1', 'k_u',
                   'hFacC', 'hFacS', 'hFacW',
                   'Depth', 'PHrefC', 'drF']

def extract_grid(ds_grid, output_dir='', prefix='llc', format='NETCDF4_CLASSIC', overwrite=False, **indexers):
    """
    ds_grid : Dataset
        A dataset with no variables, only the coordinates relative to the grid
    cond : boolean DataArray or Dataset
        The condition to make the extraction
    output_dir : str, optional
        The output directory, by default it is set to the current directory
    """
    if 'time' in indexers:
        del indexers['time']
    path = output_dir + '%s_grid.nc' %prefix
    if os.path.isfile(path) and not overwrite:
        print("File %s altready exists, skipping the extraction" %path)
    else:
        print("Extracting file %s" %path)
        ds_grid_subset = ds_grid.isel(**indexers)
        ds_grid_subset.attrs['_FillValue'] = 0.
        ds_grid_subset.to_netcdf(path, format=format)


def find_indexers(ds, lat_min=-90, lat_max=90, lon_min=0, lon_max=360):
	"""Find indexers relative to one particular region of the globe defined
	 by its latitude and longitude boundaries

	 Parameters
	 ----------
	 ds : xarray.Dataset
	    The dataset opened by using `xmitgcm.open_mdsdataset`
	 lat_min : float, optional
	    Minimum latitude
	 lat_max : float, optional
	    Maximum latitude
	 lon_min : float, optional
	    Minimum longitude
	 lon_max : float, optional
	    Maximum longitude

	Returns
	-------
	indexers :
		A dictionary of slices to be used with `xarray.Dataset.isel`
	"""
	condC = (ds.XC > lon_min) & (ds.XC < lon_max) & (ds.YC > lat_min) & (ds.YC < lat_max)
	condG = (ds.XG > lon_min) & (ds.XG < lon_max) & (ds.YG > lat_min) & (ds.YG < lat_max)
	booleanC = condC.where(condC, drop=True)
	booleanG = condG.where(condG, drop=True)
	i = booleanC.i.data
	j = booleanC.j.data.astype(int)
	i_g = booleanG.i_g.data
	j_g = booleanG.j_g.data.astype(int)
	indexers = {'i': slice(i[0], i[-1]), 'j': slice(j[0], j[-1]),
	            'i_g': slice(i_g[0], i_g[-1]), 'j_g': slice(j_g[0], j_g[-1]),
	            'face': booleanC.face.data}
	return indexers


def extract_hourly_variable(ds, var, output_dir='', prefix='llc', format='NETCDF4_CLASSIC',
                            encoding=None, overwrite=False, **indexers):
    """
	 ds : xarray.Dataset
	    The dataset opened by using `xmitgcm.open_mdsdataset` that will be converted to netCDF files
    var : str
        The name of the variable to read and store into several netCDF files
    output_dir : str, optional
        The output directory, by default it is set to the current directory
    prefix : str, optional
        The prefix used at the begining of each output file names
    format : {‘NETCDF4’, ‘NETCDF4_CLASSIC’, ‘NETCDF3_64BIT’, ‘NETCDF3_CLASSIC’}, optional
        The netCDF format
    overwrite : boolean, optional
        If True, overwrite the existing files
    **indexers : {dim: indexer, ...}
        Keyword arguments with names matching dimensions and values given by integers, slice objects or arrays.
    """
    if encoding is None:
        encoding = {'dtype': 'float32'}
    var_subset = (ds.drop(_DROPPED_COORDS).isel(**indexers))[var]
    var_subset.attrs['_FillValue'] = 0.
    ds_subset = var_subset.to_dataset()
    # General Attributes
    ds_subset.attrs['Conventions'] = "CF-1.6"
    ds_subset.attrs['source'] = "MITgcm"
    ds_subset.attrs['description'] = "NetCDF files extracted using the " \
                                     "python library xmitgcm (https://github.com/xgcm/xmitgcm)"
    # Save to the current directory if not present
    path = output_dir + '%s_%s_y%02i_m%02i_d%02i_h%02i.nc' %(prefix, var,
                                                             ds_subset['time.year'],
                                                             ds_subset['time.month'],
                                                             ds_subset['time.day'],
                                                             ds_subset['time.hour'])
    if os.path.isfile(path) and not overwrite:
        print("File %s altready exists, skipping the extraction" %path)
    else:
        print("Extracting file %s" %path)
        ds_subset.to_netcdf(path, format=format, encoding={var: encoding})
    del var_subset, ds_subset


def concatenate(ds, var, mode='monthly', output_dir='', prefix='llc', format='NETCDF4_CLASSIC', encoding=None):
	"""
	Concatenate on variable from a `xarray.Dataset` linked with several netCDF files, into hourly, monthly or
	yearly concatenated files in order to optimize the diskspace. By default, if the format is 'NETCDF4' or
	'NETCDF4_CLASSIC', a compression of level 1 is performed.

	Parameters
	----------
	ds : xarray.Dataset
		The dataset used to read the input netCDF files
	var : str
		The name of the variable to read and store into several netCDF files
	output_dir : str, optional
		The output directory, by default it is set to the current directory
	prefix : str, optional
		The prefix used at the begining of each output file names
	mode : {daily', 'monthly', 'yearly'}, optional
		Define if the files will be concatenated by day, month or year. Default is monthly.
	format : {‘NETCDF4’, ‘NETCDF4_CLASSIC’, ‘NETCDF3_64BIT’, ‘NETCDF3_CLASSIC’}, optional
		The netCDF format
	encoding : dict, optional
		Dictionary of variable specific encoding. Default is `{'zlib': True, 'complevel': 1}`
	"""
	if encoding is None:
		encoding = {'dtype': 'float32', 'zlib': True, 'complevel': 1}
	ds[var].attrs['_FillValue'] = 0.

	# Group by year and iterate on each year
	years, yearly_datasets = zip(*ds.groupby('time.year'))
	for year, ds_yearly in zip(years, yearly_datasets):

		if mode is 'yearly':
			path = output_dir + '%s_%s_y%02i.nc' % (prefix, var, year)
			ds_yearly.to_netcdf(path, format=format, unlimited_dims=['time'],
			                    encoding={var: encoding})

		else:
			# Group by month and iterate on each month
			months, monthly_datasets = zip(*ds_yearly.groupby('time.month'))
			for month, ds_monthly in zip(months, monthly_datasets):

				if mode is 'monthly':
					path = output_dir + '%s_%s_y%02i_m%02i.nc' % (prefix, var, year, month)
					ds_monthly.to_netcdf(path, format=format, unlimited_dims=['time'],
					                     encoding={var: encoding})

				elif mode is 'daily':
					# Group by day and iterate on each day
					days, daily_datasets = zip(*ds_monthly.groupby('time.day'))
					for day, ds_daily in zip(months, monthly_datasets):
						path = output_dir + '%s_%s_y%02i_m%02i_d%02i.nc' % (prefix, var, year, month, day)
						ds_daily.to_netcdf(path, format=format, unlimited_dims=['time'],
						                   encoding={var: encoding})

				else:
					raise ValueError('%s is not a valid mode' % mode)
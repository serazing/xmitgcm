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
import warnings
import numpy as np

_DROPPED_COORDS = ['dxC', 'dyC', 'dxG', 'dyG',
                   'rA', 'rAs', 'rAw', 'rAz',
                   'k_p1', 'k_u',
                   'hFacC', 'hFacS', 'hFacW',
                   'Depth', 'PHrefC', 'drF']

def _extract_grid(ds_grid, output_dir='', prefix='llc',
                  format='NETCDF4_CLASSIC',
                  overwrite=False, **indexers):
    """
    ds: Dataset
        A dataset with no variables, only the coordinates relative to the grid
    cond : boolean DataArray or Dataset
        The condition to make the extraction
    output_dir : str, optional
        The output directory, by default it is set to the current directory
    """
    new_indexers = indexers.copy()
    if 'time' in new_indexers:
        del new_indexers['time']
    grid_dir = output_dir + '/grid'
    path = grid_dir + '/%s_grid.nc' % prefix
    if not os.path.isdir(grid_dir):
		os.mkdir(grid_dir)
    if os.path.isfile(path) and not overwrite:
	    warnings.warn("Grid file %s altready exists, skipping the "
	                  "extraction" % path)
    else:
        ds_grid_subset = ds_grid.isel(**new_indexers)
        ds_grid_subset.attrs['_FillValue'] = 0.
        ds_grid_subset.to_netcdf(path, format=format)


def _extract_var(ds, var, output_dir='', prefix='llc',
                 format='NETCDF4_CLASSIC', encoding=None, overwrite=False,
                 **indexers):
	"""
	 ds : xarray.Dataset
	    The dataset opened by using `xmitgcm.open_mdsdataset` that will be
	    converted to netCDF files
    var : str
        The name of the variable to read and store into several netCDF files
    output_dir : str, optional
        The output directory, by default it is set to the current directory
    prefix : str, optional
        The prefix used at the begining of each output file names
    format : {`NETCDF4`, `NETCDF4_CLASSIC`, `NETCDF3_64BIT`,
    `NETCDF3_CLASSIC`], optional
        The netCDF format
    overwrite : boolean, optional
        If True, overwrite the existing files
    **indexers : {dim: indexer, ...}
        Keyword arguments with names matching dimensions and values given by
        integers, slice objects or arrays.
    """
	if encoding is None:
	    encoding = {'dtype': 'float32'}
	try:
		var_subset = (ds.drop(_DROPPED_COORDS).isel(**indexers))[var]
	except ValueError:
		var_subset = ds.isel(**indexers)[var]
	var_subset.attrs['_FillValue'] = 0.
	ds_subset = var_subset.to_dataset()
    # General Attributes
	ds_subset.attrs['Conventions'] = "CF-1.6"
	ds_subset.attrs['source'] = "MITgcm"
	ds_subset.attrs['description'] = ("NetCDF files extracted using the "
                                      "python library xmitgcm "
                                      "(https://github.com/xgcm/xmitgcm)")
    # Save to the current directory if not present
	var_dir = output_dir + '/' + var
	if not os.path.isdir(var_dir):
	    os.mkdir(var_dir)
	path = (var_dir +
            '/%s_%s_y%02i_m%02i_d%02i_h%02i.nc' %(prefix, var,
                                                 ds_subset['time.year'],
                                                 ds_subset['time.month'],
                                                 ds_subset['time.day'],
                                                 ds_subset['time.hour'])
			)
	if os.path.isfile(path) and not overwrite:
	    print("File %s altready exists, skipping the extraction" %path)
	else:
	    print("Extracting file %s" %path)
	    ds_subset.to_netcdf(path, format=format, encoding={var: encoding})
	del var_subset, ds_subset


def _concatenate(ds, var, mode='monthly', output_dir='', prefix='llc',
                format='NETCDF4_CLASSIC', encoding=None, overwrite=False):
	"""
	Concatenate on variable from a `xarray.Dataset` linked with several netCDF
	files, into hourly, monthly or yearly concatenated files in order to
	optimize the diskspace. By default, if the format is 'NETCDF4' or
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
	format : {`NETCDF4`, `NETCDF4_CLASSIC`, `NETCDF3_64BIT`,
    `NETCDF3_CLASSIC`], optional
	encoding : dict, optional
		Dictionary of variable specific encoding. Default is `{'zlib': True,
		'complevel': 1}`
	"""
	if encoding is None:
		encoding = {'dtype': 'float32', 'zlib': True, 'complevel': 1}
	ds[var].attrs['_FillValue'] = 0.

	# Group by year and iterate on each year
	years, yearly_datasets = zip(*ds.groupby('time.year'))
	for year, ds_yearly in zip(years, yearly_datasets):

		if mode is 'yearly':
			path = output_dir + '/%s_%s_y%04i.nc' % (prefix, var, year)
			if os.path.isfile(path) and not overwrite:
				warnings.warn("File %s altready exists, skipping the "
				              "concatenation" % path)
			else:
				ds_yearly.to_netcdf(path, format=format,
				                    unlimited_dims=['time'],
				                    encoding={var: encoding})

		else:

			yearly_dir = output_dir + '/%04i' % year
			if not os.path.isdir(yearly_dir):
				os.mkdir(yearly_dir)

			# Group by month and iterate on each month
			months, monthly_datasets = zip(*ds_yearly.groupby('time.month'))
			for month, ds_monthly in zip(months, monthly_datasets):



				if mode is 'monthly':


					path = yearly_dir + '/%s_%s_y%04i_m%02i.nc' % (prefix, var,
					                                           year, month)

					if os.path.isfile(path) and not overwrite:
						warnings.warn("File %s altready exists, skipping the "
						              "concatenation" % path)
					else:
						ds_monthly.to_netcdf(path, format=format,
						                     unlimited_dims=['time'],
						                     encoding={var: encoding})


				elif mode is 'daily':

					monthly_dir = yearly_dir + '/m%02i' % month
					if not os.path.isdir(monthly_dir):
						os.mkdir(monthly_dir)

					# Group by day and iterate on each day
					days, daily_datasets = zip(*ds_monthly.groupby('time.day'))
					for day, ds_daily in zip(days, daily_datasets):

						if not os.path.isdir(monthly_dir):
							os.mkdir(monthly_dir)
						path = (monthly_dir + '/%s_%s_y%04i_m%02i_d%02i.nc'
						        % (prefix, var, year, month, day)
						        )
						if os.path.isfile(path) and not overwrite:
							warnings.warn("File %s altready exists, skipping "
							              "the concatenation" % path)
						else:
							ds_daily.to_netcdf(path, format=format,
						                       unlimited_dims=['time'],
						                       encoding={var: encoding})

				else:
					raise ValueError('%s is not a valid mode' % mode)


def mds_to_netcdf(ds, vars, output_dir='./', prefix='llc', extract_grid=True,
                  format='NETCDF4_CLASSIC', concatenate=None, overwrite=True,
                  chunks=None, **indexers):

	if extract_grid:
		ds_grid = ds.drop([var for var in ds.data_vars])
		try:
			ds_grid = ds_grid.drop(['iter', 'time'])
		except ValueError:
			pass
		# Then extract the grid
		_extract_grid(ds_grid, output_dir=output_dir, prefix=prefix,
	                  format=format, overwrite=False, **indexers)
	else:
		grid_dir = output_dir + '/grid'
		path = grid_dir + '/%s_grid.nc' % prefix
		ds_grid = xr.open_dataset(path)
		ds = ds.assign_coords(**ds_grid.coords)
	# Then extract the variables
	for var in vars:
		for t in range(ds.time.size):
			indexers['time'] = t
			_extract_var(ds, var, output_dir=output_dir, prefix=prefix,
                         format=format, overwrite=False, **indexers)

		# Contenate files if asked
		if concatenate in ['monthly', 'daily']:
			for month in range(1, 13):
				try:
					files = output_dir + '/%s/*m%02i*.nc' %(var, month)
					ds_netcdf = xr.open_mfdataset(files, concat_dim='time',
				                                  chunks=chunks)
					_concatenate(ds_netcdf, var, mode=concatenate,
				                 output_dir=(output_dir + '/' + var),
				                 prefix=prefix, format=format,
				                 overwrite=overwrite)
					del(ds_netcdf)
				except IOError:
					pass
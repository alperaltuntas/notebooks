import sys; sys.path.append("/glade/u/home/gmarques/libs/mom6-tools/")
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
_ = xr.set_options(display_style='text')
from mom6_tools.MOM6grid import MOM6grid
from mom6_tools.m6plot import xyplot, yzplot
from mom6_tools.poleward_heat_transport import annotateObs, plotGandW, plotHeatTrans, heatTrans
import netCDF4

# Import dask
import dask

# Use dask jobqueue
from dask_jobqueue import PBSCluster

# Import a client
from dask.distributed import Client

# Setup your PBSCluster
cluster = PBSCluster(
    cores=8, # The number of cores you want
    memory='50GB', # Amount of memory
    processes=1, # How many processes
    queue='casper', # The type of queue to utilize (/glade/u/apps/dav/opt/usr/bin/execcasper)
    local_directory='$TMPDIR', # Use your local directory
    #resource_spec='select=1:ncpus=8:mem=50GB', # Specify resources
    project='P93300012', # Input your project ID here
    walltime='02:00:00', # Amount of wall time
    interface='ib0', # Interface to use
)

cluster.scale(2)
dask.config.set({'distributed.dashboard.link':'https://jupyterhub.hpc.ucar.edu/stable/user/{USER}/proxy/{port}/status'})

# MOM6 grid
base_casename = "g.e23.GMOM_JRA.TL319_t061.control.nowav.001"
archive_path = '/glade/scratch/altuntas/archive/{}/ocn/hist/'.format(base_casename)
grd = MOM6grid(archive_path+'{}.mom6.static.nc'.format(base_casename));
grd_xr = MOM6grid(archive_path+'{}.mom6.static.nc'.format(base_casename), xrformat=True);

# Auxiliary plotting functions

def open_mfdataset_minimal(full_path, variables, **kwargs):
    ds = xr.open_mfdataset(full_path, data_vars='minimal', \
                         coords='minimal', compat='override', preprocess=lambda ds:ds[variables], **kwargs)
    return ds

def summer_mean(fld):
    sm = xr.where(fld.yh<0, fld.resample(time="QS-JAN").mean()[::4].mean(dim="time"), fld.resample(time="QS-JAN").mean()[2::4].mean(dim="time"))
    sm.attrs = fld.attrs
    sm.name = "Summer means of "+fld.name
    return sm

def winter_mean(fld):
    sm = xr.where(fld.yh>0, fld.resample(time="QS-JAN").mean()[::4].mean(dim="time"), fld.resample(time="QS-JAN").mean()[2::4].mean(dim="time"))
    sm.attrs = fld.attrs
    sm.name = "Summer means of "+fld.name
    return sm

def add_season_labels(ax, season):
    if season=="winter":
        nh='JFM'; sh='JAS'
    elif season=="summer":
        nh='JAS'; sh='JFM'
    else:
        raise RuntimeError("unknown season")
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    ax.plot([grd.geolon[0,0], grd.geolon[0,-1]], [0,0], 'w--', lw=1.5)
    ax.text(25, 7, nh, ha="center", va="center", size=12, bbox=bbox_props)
    ax.text(25, -7, sh, ha="center", va="center", size=12, bbox=bbox_props)

def get_plot_title(fld):
    return "{}: {} ({})".format(fld.name, fld.attrs['long_name'], fld.attrs['units'])

def plot_field(fld, clim, zl=None, z_l=None):
    if zl is not None:
        xyplot(fld.isel(zl=zl).mean(dim='time'), grd.geolon, grd.geolat, grd.area_t, title=get_plot_title(fld), 
           colormap=plt.cm.bwr, nbins=100, clim=clim)
    elif z_l  is not None:
        xyplot(fld.isel(z_l=z_l).mean(dim='time'), grd.geolon, grd.geolat, grd.area_t, title=get_plot_title(fld), 
           colormap=plt.cm.bwr, nbins=100, clim=clim)
    else:
        raise RuntimeError("Provide zl or z_l")
        
def plot_comparison(grd, fld1, title1, fld2, title2, nmth=None, clim=None, clim_diff=None, zl=None, z_l=None, 
                    cmap=plt.cm.nipy_spectral, season=None):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(24,5))
    ax1 = ax.flatten()
    plt.suptitle(get_plot_title(fld1))

            
    if zl is not None:
        fld1 = fld1.isel(zl=zl)
        fld2 = fld2.isel(zl=zl)            
    if z_l is not None:
        fld1 = fld1.isel(z_l=z_l)
        fld2 = fld2.isel(z_l=z_l)
        
    # sub and mean
    if nmth is not None and 'time' in fld1.dims: 
        fld1 = fld1.isel(time=slice(0,nmth)).mean(dim='time')
    if nmth is not None and 'time' in fld2.dims:
        fld2 = fld2.isel(time=slice(0,nmth)).mean(dim='time')
        
        
    xyplot(np.ma.masked_invalid(fld1), grd.geolon, grd.geolat, grd.area_t, title=title1, 
       axis=ax1[0], nbins=100, colormap=cmap, clim=clim)
    xyplot(np.ma.masked_invalid(fld2), grd.geolon, grd.geolat, grd.area_t, title=title2, 
       axis=ax1[1], nbins=100, colormap=cmap, clim=clim)
    xyplot(np.ma.masked_invalid(fld1-fld2), grd.geolon, grd.geolat, grd.area_t, title='diff', 
       axis=ax1[2],nbins=40, colormap=cmap, clim=clim_diff)
    
    if season is not None:
        add_season_labels(ax1[0], season)
        add_season_labels(ax1[1], season)

def get_clim(fld):
    maxval = fld.max().data.compute()
    minval = fld.min().data.compute()
    
    if minval <= 0.0 and maxval >= 0.0:
        absmax = max(-minval, maxval)
        return (-absmax, absmax)
    else:
        return (minval, maxval)
    
def get_heat_transport_obs():
    """Plots model vs obs poleward heat transport for the global, Pacific and Atlantic basins"""
    # Load Observations
    fObs = netCDF4.Dataset('/glade/work/gmarques/cesm/datasets/Trenberth_and_Caron_Heat_Transport.nc')
    #Trenberth and Caron
    yobs = fObs.variables['ylat'][:]
    NCEP = {}; NCEP['Global'] = fObs.variables['OTn']
    NCEP['Atlantic'] = fObs.variables['ATLn'][:]; NCEP['IndoPac'] = fObs.variables['INDPACn'][:]
    ECMWF = {}; ECMWF['Global'] = fObs.variables['OTe'][:]
    ECMWF['Atlantic'] = fObs.variables['ATLe'][:]; ECMWF['IndoPac'] = fObs.variables['INDPACe'][:]

    #G and W
    Global = {}
    Global['lat'] = np.array([-30., -19., 24., 47.])
    Global['trans'] = np.array([-0.6, -0.8, 1.8, 0.6])
    Global['err'] = np.array([0.3, 0.6, 0.3, 0.1])

    Atlantic = {}
    Atlantic['lat'] = np.array([-45., -30., -19., -11., -4.5, 7.5, 24., 47.])
    Atlantic['trans'] = np.array([0.66, 0.35, 0.77, 0.9, 1., 1.26, 1.27, 0.6])
    Atlantic['err'] = np.array([0.12, 0.15, 0.2, 0.4, 0.55, 0.31, 0.15, 0.09])

    IndoPac = {}
    IndoPac['lat'] = np.array([-30., -18., 24., 47.])
    IndoPac['trans'] = np.array([-0.9, -1.6, 0.52, 0.])
    IndoPac['err'] = np.array([0.3, 0.6, 0.2, 0.05,])

    GandW = {}
    GandW['Global'] = Global
    GandW['Atlantic'] = Atlantic
    GandW['IndoPac'] = IndoPac
    return NCEP, ECMWF, GandW, yobs

NCEP, ECMWF, GandW, yobs = get_heat_transport_obs()

def get_adv_diff(ds):
    # create a ndarray subclass
    class C(np.ndarray): pass

    varName = 'T_ady_2d'
    if varName in ds.variables:
        tmp = np.ma.masked_invalid(ds[varName].values)
        tmp = tmp[:].filled(0.)
        advective = tmp.view(C)
        advective.units = 'W'
    else:
        raise Exception('Could not find "T_ady_2d"')

    varName = 'T_diffy_2d'
    if varName in ds.variables:
        tmp = np.ma.masked_invalid(ds[varName].values)
        tmp = tmp[:].filled(0.)
        diffusive = tmp.view(C)
        diffusive.units = 'W'
    else:
        diffusive = None
        warnings.warn('Diffusive temperature term not found. This will result in an underestimation of the heat transport.')

    varName = 'T_lbm_diffy'
    if varName in ds.variables:
        tmp = np.ma.masked_invalid(ds_sel[varName].sum('z_l').values)
        tmp = tmp[:].filled(0.)
        diffusive = diffusive + tmp.view(C)
    else:
        warnings.warn('Lateral boundary mixing term not found. This will result in an underestimation of the heat transport.')

    return advective, diffusive


def plot_heat_trans(ds, label, linestyle='-'):
    adv, diff = get_adv_diff(ds)
    HT = heatTrans(adv,diff); y = ds.yq
    plt.plot(y, HT, linewidth=3, linestyle=linestyle, label=label)
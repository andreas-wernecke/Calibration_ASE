# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 11:05:20 2018

@author: andi
to be run on a machine with amr libs installed (laptop)
"""
import gdal
import os, sys, fnmatch
from os.path import isfile, join
home=os.getenv('HOME')
genpath='/home/andi/BISICLES/libamrfile/python/AMRFile'
#libpath="/home/andi/BISICLES_H/BISICLES/code/libamrfile"
#if not sys.path.__contains__(genpath):
#    sys.path.append(genpath)
sys.path.append(genpath)

from amrfile import io as amrio
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.colors as col
from mpl_toolkits.basemap import Basemap

test=0
years=7.

testdatadir = "/home/andi/BISICLES/testdata/"
datadir = "/home/andi/BISICLES/data/"
yeardir="ensembleB_07_decompressed/"
#read tab
tabfn="parameterValues.tab"
tabf=open(testdatadir+tabfn, 'r')
hline= tabf.readline()#header
runn, V1, V2, V3 = [], [], [], []
for line in tabf.readlines():
    tmp =line.rsplit(' ')
    runn.append(tmp[1])
    V1.append(tmp[2]) #traction
    V2.append(tmp[3]) #viscosity
    V3.append(tmp[4]) #melt rate
runn, V1, V2, V3 = np.asarray(runn), np.asarray(V1, dtype='float'), \
    np.asarray(V2, dtype='float'), np.asarray(V3, dtype='float')
Vs=np.vstack([V1, V2, V3])
tabf.close()

#read maps
if test:
    fns=fnmatch.filter(os.listdir(datadir), "plot*.hdf5")
    fns.remove("plot.extract.ase-lhs-B0000-Cthird-mb.4lev.000000.2d.hdf5")
else:
    fns=[]
    Bns=os.listdir(datadir+yeardir)
    for Bn in Bns:
        for fn_inBn in os.listdir(datadir+yeardir+Bn):
            fns.append(Bn+'/'+fn_inBn)
        
#plot.amundsen.2d.hdf5
thkcomp = "thickness"
#thkcomp = "Z_base"
#"xVel", "yVel", "Z_surface", "Z_bottom", "Z_base", "Vel_magnitude"
#"Mesh", "levels", "patches", "Vel"

#read a box of thickness data at the lowest resolution

namr=len(fns)
Vs_full=np.zeros([5, namr])#
Y=np.zeros([57344,namr])

level = 0
order = 0 # interpolation order, 0 for piecewise constant, 1 for linear
amrID = amrio.load(testdatadir+"plot.extract.ase-lhs-B0000-Cthird-mb.4lev.000000.2d.hdf5")
lo,hi = amrio.queryDomainCorners(amrID, level)
x0,y0, bottom0 = amrio.readBox2D(amrID, level, lo, hi, "Z_bottom", order)
x0,y0, surface0 = amrio.readBox2D(amrID, level, lo, hi, "Z_surface", order)
thk_t0 = surface0-bottom0
amrio.free(amrID)
print datadir+yeardir+fns[0]

goodruns=np.hstack([range(24), range(25,140), range(141,namr)])
for i in goodruns:
    print i
    amrID = amrio.load(datadir+yeardir+fns[i])#fns0=start
    if thkcomp=="thickness":
        x0,y0, bottom0 = amrio.readBox2D(amrID, level, lo, hi, "Z_bottom", order)
        x0,y0, surface0 = amrio.readBox2D(amrID, level, lo, hi, "Z_surface", order)
        thk0 = surface0-bottom0
    else:
        x0,y0,thk0 = amrio.readBox2D(amrID, level, lo, hi, thkcomp, order)
        
    Y[:,i]=thk0.flatten()
    tmp0, tmp1, amrIDrun, tmp3, amrC, amrBed = fns[i].rsplit('-')
    #print amrIDrun
    Vs_full[:3,i]=Vs[:, np.nonzero(runn=='"'+amrIDrun+'"')[0][0]]
    Vs_full[3,i] = amrC == 'Cone'
    Vs_full[4,i] = amrBed[:4] == 'zero'
    amrio.free(amrID)
    
    
#convert to dhdt + Row centering
dhdt=np.subtract(Y.T, thk_t0.flatten()).T/years

dhdt_mean_empi=dhdt.mean(axis=1)
dhdt_var = dhdt.var(axis=1)
dhdt_c=np.subtract(dhdt.T, dhdt_mean_empi).T
if 1:
    np.save("/home/andi/Dropbox/mypaper/dhdt_centered_{:02d}_v001.npy".format(int(years)), dhdt_c)
    np.save("/home/andi/Dropbox/mypaper/dhdt_mean_{:02d}_v001.npy".format(int(years)), dhdt_mean_empi)
    np.save("/home/andi/Dropbox/mypaper/Vs_{:02d}_v001.npy".format(int(years)), Vs_full)
    np.save("/home/andi/Dropbox/mypaper/xy_{:02d}_v001.npy".format(int(years)), [x0, y0])

#for i in range(10):
#    Yc[:,i]=Y[:,i]-y_mean_empi
#
#u,s,v = np.linalg.svd(Yc, full_madtrices=False)

#plt.figure()
#plt.imshow(dhdt_c[:,1].reshape([256,224]))
#plt.colorbar()


m=Basemap(width=5400000., height=5400000., projection='stere',\
          ellps='WGS84', lon_0=180., lat_0=-90., lat_ts=-71., resolution='i')
x_sp, y_sp=m(-90,-90, inverse=True)

if 1:
    print('Loading Bedmap2')
    datadir_b='/home/andi/work/bedmap/bedmap2_tiff/'
    
    fn_b='bedmap2_bed.tif'
    #fn_b='bedmap2_thickness_uncertainty_5km.tif' 
    #fn_b='bedmap2_rockmask.tif'
    ds=gdal.Open(datadir_b+fn_b)
    h_b=ds.ReadAsArray()
    mask_hb=h_b==32767
    h_b=np.ma.array(h_b, mask=mask_hb)
    
    fn_b='bedmap2_thickness.tif' 
    ds=gdal.Open(datadir_b+fn_b)
    s_b=ds.ReadAsArray()
    mask_sb=s_b==32767
    s_b=np.ma.array(s_b, mask=mask_sb)
    
    fn_b='bedmap2_icemask_grounded_and_shelves.tif'
    ds=gdal.Open(datadir_b+fn_b)
    mask=ds.ReadAsArray()

    x_bcenter, y_bcenter = np.meshgrid(np.linspace(-3333000, 3333000, 6667), \
        np.linspace(3333000, -3333000, 6667)) #map=projection
    lonb, latb=m(x_bcenter, y_bcenter, inverse=True)
    #xbn, ybn=m(lonb, (latb+90.)*1.027-90.)#wrong geoid?
    
    #reduce resolution to 4X4 km
    s_blow=h_b[2:-1, 2:-1].reshape([1666,4, 1666, 4]).mean(axis=3).mean(axis=1)
    mask_blow=mask[2:-1, 2:-1].reshape([1666,4, 1666, 4]).min(axis=3).min(axis=1)
    x_blow=x_bcenter[2:-1, 2:-1].reshape([1666,4, 1666, 4]).mean(axis=3).mean(axis=1)
    y_blow=y_bcenter[2:-1, 2:-1].reshape([1666,4, 1666, 4]).mean(axis=3).mean(axis=1)
    
    xoff, yoff = 64000., 116000.
    mask_area=np.logical_and(np.logical_and(np.logical_and(x_blow>-1894000+xoff, \
        x_blow<=-998000+xoff), y_blow>-1019000+yoff), y_blow<=5000+yoff)
    #s_cut=np.ma.array(s_blow, mask=mask_area==0)
    #x_cut=np.ma.array(x_blow, mask=mask_area==0)    
    #y_cut=np.ma.array(y_blow, mask=mask_area==0)
    s_cut=s_blow[mask_area].reshape([256, 224])   
    x_cut=x_blow[mask_area].reshape([256, 224])
    y_cut=y_blow[mask_area].reshape([256, 224])
    
    if 0:
        plt.figure()
        x_blow, y_blow = x_blow+x_sp, y_blow+y_sp 
        x_cut, y_cut = x_cut+x_sp, y_cut+y_sp 
        #m.pcolormesh(x_blow, y_blow, s_blow)
        m.pcolormesh(x_cut, y_cut, s_cut)
        ax=plt.gca()
        ax.set_xlim([np.min(x_cut), np.max(x_cut)])
        ax.set_ylim([np.min(y_cut), np.max(y_cut)])
        plt.colorbar()
    
        plt.figure()
        m.pcolormesh(x0, y0, bottom0)
        plt.colorbar()
        ax=plt.gca()
        ax.set_xlim([np.min(x0), np.max(x0)])
        ax.set_ylim([np.min(y0), np.max(y0)])
        
        plt.figure()
        m.pcolormesh(x0, y0, bottom0-s_cut[::-1,:])
        plt.colorbar()
        ax=plt.gca()
        ax.set_xlim([np.min(x0), np.max(x0)])
        ax.set_ylim([np.min(y0), np.max(y0)])


    np.save("/home/andi/Dropbox/mypaper/xy_estimate_v001.npy", [x_cut, y_cut])

if 0:
    for i in range(4):
        for j in range(4):
            i=2
            j=2
            if j==3:
                if i==3:
                    s_blow=h_b[3:, 3:].reshape([1666,4, 1666, 4]).mean(axis=3).mean(axis=1)
                    x_blow=x_bcenter[3:, 3:].reshape([1666,4, 1666, 4]).mean(axis=3).mean(axis=1)
                    y_blow=y_bcenter[3:, 3:].reshape([1666,4, 1666, 4]).mean(axis=3).mean(axis=1)
                else:
                    s_blow=h_b[i:-3+i, 3:].reshape([1666,4, 1666, 4]).mean(axis=3).mean(axis=1)
                    x_blow=x_bcenter[i:-3+i, 3:].reshape([1666,4, 1666, 4]).mean(axis=3).mean(axis=1)
                    y_blow=y_bcenter[i:-3+i, 3:].reshape([1666,4, 1666, 4]).mean(axis=3).mean(axis=1)
            else:
                if i==3:
                    s_blow=h_b[:-3, j:-3+j].reshape([1666,4, 1666, 4]).mean(axis=3).mean(axis=1)
                    x_blow=x_bcenter[:-3, j:-3+j].reshape([1666,4, 1666, 4]).mean(axis=3).mean(axis=1)
                    y_blow=y_bcenter[:-3, j:-3+j].reshape([1666,4, 1666, 4]).mean(axis=3).mean(axis=1)
                else:
                    s_blow=h_b[i:-3+i, j:-3+j].reshape([1666,4, 1666, 4]).mean(axis=3).mean(axis=1)
                    x_blow=x_bcenter[i:-3+i, j:-3+j].reshape([1666,4, 1666, 4]).mean(axis=3).mean(axis=1)
                    y_blow=y_bcenter[i:-3+i, j:-3+j].reshape([1666,4, 1666, 4]).mean(axis=3).mean(axis=1)
    
            xoff, yoff = 64000., 116000.
            mask_area=np.logical_and(np.logical_and(np.logical_and(x_blow>-1894000+xoff, \
                x_blow<=-998000+xoff), y_blow>-1019000+yoff), y_blow<=5000+yoff)
            s_cut=s_blow[mask_area].reshape([256, 224])  
        
            plt.figure()
            print i+10.*j
            m.pcolormesh(x0, y0, bottom0-s_cut[::-1,:])
            plt.colorbar()
            ax=plt.gca()
            ax.set_xlim([np.min(x0), np.max(x0)])
            ax.set_ylim([np.min(y0), np.max(y0)])
    

#%%


























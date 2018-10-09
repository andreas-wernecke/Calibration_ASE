# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 12:11:45 2018

@author: andi
"""
#import gdal
import numpy as np
import os, sys#, fnmatch
home=os.getenv('HOME')
genpath=home+'/home_zmaw/sync/seaice_remote_sensing/general/'
sys.path.append(genpath)
import ant_tools as antt
from scipy.ndimage.interpolation import zoom
#from scipy.interpolate import griddata
import GPy
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
#import pymc3 as mc
#import MCMC

def setup_emulator(sim_in, sim_out, s='none'):
    """Sets up an gaussian process emulator for a given set of n output values sim_out [n]
     for a set of input values sim_in [n x m] where m is the dimension of inputs.
    
    Hyperparameters are optimized at marginal likelyhood with 10 restarts of optimization.
    
    Output: A GPy Gaussian process model, the kernel/covarianze function and
    mean function.
    """
    if np.shape(sim_in)[0]!=len(sim_out):
        if np.shape(sim_in)[1]==len(sim_out):
            sim_in=sim_in.T
        else: print('Wrong Shape!')
    dim=np.shape(sim_in)[1]
    if np.shape(np.shape(sim_out))[0]!=2:
        sim_out=sim_out.reshape([-1,1])
    if s=='none': s=1.
    lengthscales=np.ones(dim)*0.5
    #gpk=GPy.kern.RBF(input_dim=dim, variance=var, lengthscale=lengthscales, ARD=True)
    #gpk=GPy.kern.Exponential(input_dim=dim, variance=var, lengthscale=lengthscales, ARD=True)
    gpk=GPy.kern.Matern52(input_dim=dim, variance=s, lengthscale=lengthscales, ARD=True)
    meanf = GPy.mappings.Constant(input_dim=dim, output_dim=1)
    offset_prior = GPy.priors.Gaussian(mu=0, sigma=0.5)
    #meanf = GPy.mappings.Additive(GPy.mappings.Linear(input_dim=5,output_dim=1),\
    #                                    GPy.mappings.Constant(input_dim=5, output_dim=1))
    gpm=GPy.models.GPRegression(sim_in, sim_out, gpk, noise_var=0., mean_function=meanf)
    gpm.mean_function.set_prior(offset_prior)
    #gpm.constrain_positive()#are constrained anyways
    gpm.Gaussian_noise.variance.fix(0.)
    #gpm.rbf.lengthscale.constrain_bounded(lower=1e-10, upper=2)
    gpm.kern.lengthscale.constrain_bounded(1e-10,4)
    #gpm['constmap.C']=1
    gpm.optimize_restarts(2, messages=1)
    #gpm.optimize(messages=0)
    
    print(gpm[''])
    return gpm, gpk, meanf
    
    
def predict_PC(u, s, vemu, emu_in):
    v_vec, var_v_vec=predictXk(vemu, emu_in)
    ddhdt = u.dot(v_vec*s)
    return ddhdt
    
def predictXk(vemu, emu_in):
    if np.shape(np.shape(emu_in))[0]==1:
        emu_in=emu_in.reshape([1,-1])
    k=len(vemu)
    v_vec, var_v_vec = np.zeros(k), np.zeros(k)
    for i in range(k):
        v_vec[i], var_v_vec[i] = vemu[i][0].predict(emu_in)
    return v_vec, var_v_vec

#def all the parts of Chang2016 eq 17 below:
def f_sigd(sigd):
    a=3 # if you change this also change gamma(a)=2 below
    b=1.
    return b**a/2.*sigd**(-a-1)*np.exp(-b/sigd)

def f_theta(theta):
    bounds=np.array([[0,1],
                    [0,1],
                    [0,1]])
    for i in range(len(theta)):
        if theta[i]<bounds[i,0] or theta[i]>bounds[i,1]: return 0.
    return 1.
    
def f_v(v, sigd):
    return np.exp(-(v-1.)**2/(2.*sigd**2))/sigd

def f_eta(eta, PC_emus, theta):#works but is extremely narrow!
    if np.shape(np.shape(theta))[0]==1: theta=theta.reshape([1,-1])
    if len(eta)!=len(PC_emus): print('STOP')
    fs=np.zeros_like(eta)
    for i in range(len(eta)):
        mu, sig = PC_emus[i][0].predict(theta)
        fs[i]=np.exp(-(eta[i]-mu)**2/(2.*sig**2))/sig
    return np.prod(fs)

def f_Z(Z, eta, v, dhdt_mean, ur, sr, disc, Z_err=1.):
    #Z_err (obs uncertanty (std)) can be spatially varying or set to 1 verywhere (default)
    #not sure what to do with s! Ithink=const to have all PC being equally effected bycal
    if np.shape(np.shape(Z_err))[0]==0: Z_err=np.ones_like(Z)*Z_err
    lam=dhdt_mean + ur.dot(sr*eta) + disc*v
    lam=model2obsgrid(lam)
    if 0:
        print(np.sum(lam))
        plt.figure()
        plt.imshow(Z-lam)
        plt.colorbar()
    fs=np.exp(-(Z-lam)**2/(2.*Z_err**2))#/Z_err maybe add 1/Zerr back later
    return np.prod(np.prod(fs))
    #return np.sum(fs)

def model2obsgrid(field):
    #we are very quick and dirty here!
    field=zoom(field.reshape([256,224]), 2, order=1) #this works slidly unexpected
    field=np.ma.array(field, mask=field==0) #I mean:this one is obvious ;)
    field=field[2:, :-3].reshape([102, 5, 89, 5]).mean(axis=1).mean(axis=2)# ~500m off
    return field[19:-11,9:-26]
    
#%%%


datadir= "/home/dusch/Dropbox/mypaper/"
#datadir= "/home/andi/Dropbox/mypaper/"
dhdt=np.load(datadir+"dhdt_centered.npy")
dhdt_mean=np.load(datadir+"dhdt_mean.npy")
#x0, y0 = np.load(datadir+"xy.npy")
#x, y = np.meshgrid(x0, y0)
x, y = np.load(datadir+"xy_estimate_v001.npy")
Vs=np.load(datadir+"Vs.npy")

for i in range(np.shape(dhdt)[1]):
    print(np.sum(dhdt_mean+dhdt[:,i]))

u,s,v = np.linalg.svd(dhdt, full_matrices=False)
#dhdt=u.dot(np.diag(s)).dot(v)
#u:PCs, s:sigular values (importance of pc); v:connets Vs to PCs
#v to be replaced by emulators
#set k=5 for now, si/sum(s)<0.05 for i>k

k=10
ur,sr,vr = u[:,:k], s[:k], v[:k,:]

PC_emus=[]
for i in range(k):
    PC_emus.append(setup_emulator(Vs, vr[i,:], s='none'))


#LOO Validation
if 0:
    n_out=np.shape(vr)[1]
    LOO_mu, LOO_var = np.zeros(n_out), np.zeros(n_out)
    for out in range(4):#n_out):#this is the leave one out loop
        #time.sleep(5.5)
        Vs_cut=np.hstack([Vs[:,:out], Vs[:,out+1:]])#reduced input
        vr_cut=np.hstack([vr[:,:out], vr[:,out+1:]])#and output
        #gpm, gpk, meanf = setup_emulator(X_cut, ddat_cut, year)
        #gpm.constrain_positive()#are constrained anyways
        #gpm.Gaussian_noise.variance.fix(0.)
        #gpm.rbf.lengthscale.constrain_bounded(lower=1e-10, upper=2)
        #using the previously optimized hyperparameters (uncomment to optimze on the run)
        #gpm.optimize_restarts(10, messages=0)
        #and doing+saving the LOO predictions
        PC_emus_cut=[]
        for i in range(k):
            PC_emus_cut.append(setup_emulator(Vs_cut, vr_cut[i,:], s='none'))
    
        dhdt_cut = predict_PC(ur, sr, PC_emus_cut, Vs[:,out])
    
        #dhdt
        if 1:
            fig, axes = plt.subplots(1,2)
            axes[0].pcolormesh(x, y, dhdt_cut.reshape([256,224]), vmin=-10, vmax=10)
            pcol=axes[1].pcolormesh(x, y, dhdt[:,out].reshape([256, 224]), vmin=-10, vmax=10)
            plt.colorbar(pcol, ax=axes[1])
            #axes[0].
        #LOO_mu[out], LOO_var[out] = gpm.predict(X[out,:].reshape(1,-1))
        #print "Predicted: " + str(LOO_mu[out])
        #print "Simulated: " + str(ddat[year,out])
        #print gpm.rbf.lengthscale

if 1: #fake obs from central run with noise
    central_r=np.nonzero(np.logical_and(Vs[0]==0.5, np.logical_and(Vs[1]==0.5, Vs[2]==0.5)))[0][0]
    
    if 1:
        #GP to produce "noise"
        lengthscales=np.ones(1)*100000
        gpk=GPy.kern.Matern52(input_dim=2, variance=1., lengthscale=lengthscales)
        gpm=GPy.models.GPRegression(np.array([[-1e5,-1e5]]), np.array([[0]]), gpk, noise_var=0)
        obs_noise=gpm.posterior_samples_f(np.vstack([x[::4,::4].flatten(), y[::4,::4].flatten()]).T, full_cov=True, size=1)
        #obs_noise=gpm.predict(np.vstack([x.flatten(), y.flatten()]).T)
    
        obs=dhdt[:,central_r].reshape([256, 224])+0.5*zoom(obs_noise.reshape([64, 56]), 4., order=3)
        nomodel=np.min(dhdt, axis=1)==np.max(dhdt, axis=1)
        obs=np.ma.array(obs, mask=nomodel)
        np.save(datadir+'fake_obs.npy', obs.data)
        np.save(datadir+'fake_obsm.npy', obs.mask)
    else:
        obs=np.ma.array(np.load(datadir+'fake_obs.npy'), mask=np.load(datadir+'fake_obsm.npy'))
    obs=obs+dhdt_mean.reshape([256,224])
#if 0:#Helm eta al. DEM, not change
#    cspath='/home/andi/Downloads/'
#    csfn='DEM_ANT_CS_20130901.tif'#https://doi.pangaea.de/10.1594/PANGAEA.831392
#    ds = gdal.Open(cspath+csfn)
#
#    cs=ds.ReadAsArray()
#	#cs=cs*916./(1029.-916.)
#    rows=ds.RasterXSize
#    cols=ds.RasterYSize
#    csx = np.linspace(m.llcrnrx, m.urcrnrx, rows)#map is equal to projection
#    csy = np.linspace(m.llcrnrx, m.urcrnrx, cols)
#    m_csx=np.logical_and(-1.67e6 < csx, csx < -1.33e6) #KGVI
#    m_csy=np.logical_and(-4.27e6 < csy, csy < -4.e6)
#    m_cs2d1, m_cs2d2 = np.meshgrid(m_csx, m_csy)
#    m_cs2d=np.logical_and(m_cs2d1, m_cs2d2)
#    cs_tmp=cs[::-1][m_cs2d].reshape(sum(m_csy), sum(m_csx))
#    csx2d, csy2d=np.meshgrid(csx[m_csx], csy[m_csy])
#    geofield=griddata(np.array([geox, geoy]).T, geoha, np.array([csx2d.flatten(), csy2d.flatten()]).T, method='nearest').reshape(sum(m_csy), sum(m_csx))
#    csfb=cs_tmp-geofield
#    csth=csfb*1029./(1029.-916.)
#    #imcs=m.imshow(cs[::-1][, ax=ax, vmin=-40, vmax=40,cmap='bwr')#vmin, vmax=vmax)
#    imcs=m.pcolormesh(csx2d, csy2d, csth, vmin=vmin, vmax=vmax)
#    #m.pcolormesh(csx, csy, cs[::-1], vmin=vmin, vmax=vmax) #slower, but shows the same, so csx/csy (or csxy[::-1]) can be used to connect with OIB data
#    plotlines(m, ax, path='/home/dusch/home_zmaw/phd/programme/surfacetypes/')
#    plt.colorbar(imcs, ax=ax)


#model discrepancy
#first try: min distance between (dim reduced) ensemble members and obs where obs are not "sourounded" 

dhdt_dimred=ur.dot(np.diag(sr)).dot(vr)
dist=dhdt_dimred.T-obs.flatten()
dist=dist.T
obs_notin_ensemble=np.logical_or(np.sum(np.sign(dist), axis=1)==np.shape(dist)[1], np.sum(np.sign(dist), axis=1)==-1*np.shape(dist)[1])
disc=np.zeros(57344)
disc=np.where(np.sum(np.sign(dist), axis=1)==np.shape(dist)[1], np.min(dist,axis=1), disc)
disc=np.where(np.sum(np.sign(dist), axis=1)==-1*np.shape(dist)[1], np.max(dist,axis=1), disc)
disc=np.ma.array(disc, mask=obs.mask)



if 0:#test f_Z and MCMC
    f_Z(obs.flatten(), 
        [-0.0311924,-0.07719354,-0.00921586,0.08950857,0.31253308], 
        1.,
        dhdt_mean,
        ur, 
        sr, 
        disc, 
        Z_err=1.)
        
    postm=mc.Model()
    
    with postm:
        sigd=mc.InverseGamma('sigd', alpha=3, beta=1)
        theta=mc.Uniform('theta', lower=0, upper=1, shape=3)
        v=mc.Normal('v', mu=1, sd=sigd)
        
        fs=np.zeros(5)
        for i in range(5):
            mu, sig = PC_emus[i][0].predict(theta)#             dosent work :(
            fs[i]=mc.Normal(mu=mu, sd=sig)
        f_eta2=fs[0]*fs[1]*fs[2]*fs[3]*fs[4]
        
        #def f_Z(Z, eta, v, dhdt_mean, ur, sr, disc, Z_err=1.):
            #everything needs to be on the same grid (x,y), Z can be masked but needs to be gridded
            #Z_err (obs uncertanty (std)) can be spatially varying or set to 1 verywhere (default)
            #not sure what to do with s!
        Z_err=1.
        if np.shape(np.shape(Z_err))[0]==0: Z_err=np.ones_like(obs)*Z_err
        lam=dhdt_mean + ur.dot(sr*f_eta2) + disc*v
        fs_obs=mc.Normal('fs_obs', mu=lam, sd=Z_err, observed=obs)

if 0:
    import MCMC
    #hand written, see other file
if 0: 
    import pymc
    import numpy as np

    # Some data
    n = 5*np.ones(4,dtype=int)
    x = np.array([-.86,-.3,-.05,.73])

    # Priors on unknown parameters
    alpha = pymc.Normal('alpha',mu=0,tau=.01)
    beta = pymc.Normal('beta',mu=0,tau=.01)

    # Arbitrary deterministic function of parameters
    @pymc.deterministic
    def theta(a=alpha, b=beta):                             #make this f_eta!!!
        """theta = logit^{-1}(a+b)"""
        return pymc.invlogit(a+b*x)

    # Binomial likelihood for data
    d = pymc.Binomial('d', n=n, p=theta, value=np.array([0.,1.,3.,5.]),\
                  observed=True)

    S = pymc.MCMC(mymodel, db='pickle')
    S.sample(iter=10000, burn=5000, thin=2)
    pymc.Matplot.plot(S)

    #https://pymc-devs.github.io/pymc/README.html#features
    #and
    #http://sdsawtelle.github.io/blog/output/mcmc-in-python-with-pymc.html


#%%%
#calibration data
from netCDF4 import Dataset

with Dataset(datadir+'dhdt_10km_sig0p5_0p0.nc') as f:
    x_obs=f.variables['x'][:]
    y_obs=f.variables['y'][:]
    year_obs=f.variables['t'][:]
    dhdt_obs=f.variables['dhdt'][:,:,:]
dhdt_obs.mask=np.isnan(dhdt_obs.data)

x_obs, y_obs = np.meshgrid(x_obs, y_obs)

#match obs (10km) and model (4km) by interpolate model to 2 and take 5X5 mean
#lets do this with the PC to speed things up

dhdt2km=zoom((dhdt_mean+dhdt[:,central_r]).reshape([256,224]), 2, order=1)
dhdt10km=dhdt2km[:510,:445].reshape([102,5,89,5]).mean(axis=1).mean(axis=2)
dhdt10km=np.ma.array(dhdt10km, mask=dhdt10km==0)
dhdt2km=np.ma.array(dhdt2km, mask=dhdt2km==0)

x2km=zoom(x.reshape([256,224]), 2, order=1)
x10km=x2km[:510,:445].reshape([102,5,89,5]).mean(axis=1).mean(axis=2)

y2km=zoom(y.reshape([256,224]), 2, order=1)
y10km=y2km[:510,:445].reshape([102,5,89,5]).mean(axis=1).mean(axis=2)

#define basemap and bring obs+model to it
m=Basemap(width=5400000., height=5400000., projection='stere',\
          ellps='WGS84', lon_0=180., lat_0=-90., lat_ts=-71., resolution='i')
                              
x_sp, y_sp = m(0, -90)

xbase, ybase=-1*x2km+x_sp, -1*y2km[::-1,:]+y_sp #transform
xbase=xbase[2:, :-3].reshape([102, 5, 89, 5]).mean(axis=1).mean(axis=2)#reduce res
ybase=ybase[2:, :-3].reshape([102, 5, 89, 5]).mean(axis=1).mean(axis=2)
dhdt2km=dhdt2km[2:, :-3].reshape([102, 5, 89, 5]).mean(axis=1).mean(axis=2)

xbase=xbase[19:-11,9:-26]#cut to area
ybase=ybase[19:-11,9:-26]#cut to area
dhdtbase=dhdt2km[19:-11,9:-26]

xobsbase, yobsbase = -1*x_obs*1000.+x_sp, -1*y_obs*1000+y_sp
#xbase, ybase are now = xobsbase, yobsbase

if 1:
    fig, ax=plt.subplots()
    m.pcolormesh(xbase, ybase, dhdtbase, cmap='plasma', vmin=-8, vmax=0)
    cbar=plt.colorbar()
    cbar.set_label('Surface Elevation Change [m/a]', fontsize=18)
    antt.plotlines(m, ax=ax, path='/home/dusch/home_zmaw/phd/programme/surfacetypes/')
    ax.set_xlim([3.9e6, 4.4e6])
    ax.set_ylim([2.7e6, 3.4e6])
if 1:
    fig, ax=plt.subplots()
    m.pcolormesh(xobsbase, yobsbase, model2obsgrid(obs),cmap='plasma', vmin=-8, vmax=0)    
    #plt.imshow(dhdt_mean.reshape([256,224]))
    #plt.imshow(dhdt10km[20:-10,10:-25])
    #plt.imshow(dhdt10km[20:-10,10:-25]==0) 
    cbar=plt.colorbar()
    cbar.set_label('Surface Elevation Change [m/a]', fontsize=18)
    antt.plotlines(m, ax=ax, path='/home/dusch/home_zmaw/phd/programme/surfacetypes/')
    ax.set_xlim([3.9e6, 4.4e6])
    ax.set_ylim([2.7e6, 3.4e6])
    
    fig, ax=plt.subplots()
    #m.pcolormesh(xobsbase, yobsbase, model2obsgrid(obs)-dhdtbase ,cmap='seismic', vmin=-4, vmax=4)    
    m.pcolormesh(xobsbase, yobsbase, dhdt_obs[-2,:,:]-dhdtbase ,cmap='seismic', vmin=-4, vmax=4)   
    #plt.imshow(dhdt_mean.reshape([256,224]))
    #plt.imshow(dhdt10km[20:-10,10:-25])
    #plt.imshow(dhdt10km[20:-10,10:-25]==0) 
    cbar=plt.colorbar()
    cbar.set_label('Elevation Change Difference [m/a]', fontsize=18)
    antt.plotlines(m, ax=ax, path=home+'/home_zmaw/phd/programme/surfacetypes/')    
    ax.set_xlim([3.9e6, 4.4e6])
    ax.set_ylim([2.7e6, 3.4e6])
    
sses=np.ones(k)
for i in range(np.shape(Vs)[1]):
    #print(np.sum(predict_PC(ur, sr, PC_emus, Vs[:,i])))    
    PC_cons, PC_cons_var = predictXk(PC_emus, Vs[:,i])
    L=f_Z(dhdt_obs[-2,:,:], eta=PC_cons, v=0, dhdt_mean=dhdt_mean, ur=ur, sr=sr, disc=0, Z_err=5.)
    #L=f_Z(model2obsgrid(obs), eta=PC_cons, v=0, dhdt_mean=dhdt_mean, ur=ur, sr=sr, disc=0, Z_err=5.)
    print(L)

Ls=np.zeros([11,11, 11])
for i, v1 in enumerate(np.linspace(0,1,11)):
    for j, v2 in enumerate(np.linspace(0,1.,11)):
        for kk, v3 in enumerate(np.linspace(0.,1.,11)):
            #print(np.sum(predict_PC(ur, sr, PC_emus, Vs[:,i])))    
            PC_cons, PC_cons_var = predictXk(PC_emus, np.array([v1,v2,v3]))
            Ls[i,j,kk]=f_Z(dhdt_obs[-2,:,:], eta=PC_cons, v=0, dhdt_mean=dhdt_mean, ur=ur, sr=sr, disc=0, Z_err=8.)
            #Ls[i,j,kk]=f_Z(model2obsgrid(obs), eta=PC_cons, v=0, dhdt_mean=dhdt_mean, ur=ur, sr=sr, disc=0, Z_err=8.)
            #print(L)
        
        
fig, ax=plt.subplots()
plt.pcolor(np.linspace(-0.05,1.05,12), np.linspace(-0.05,1.05,12),Ls.sum(axis=2).T/np.sum(Ls))
ax.set_xlabel('Traction', fontsize=18)
ax.set_ylabel('Viscosity', fontsize=18)
cbar=plt.colorbar()
cbar.set_label('Likelyhood', fontsize=18)
ax.set_ylim([0,1.])
ax.set_xlim([0,1.])
#ax.set_ylim([0,1.])

fig, ax=plt.subplots()
plt.pcolor(np.linspace(-0.05,1.05,12), np.linspace(-0.05,1.05,12),Ls.sum(axis=1).T/np.sum(Ls))
ax.set_xlabel('Traction', fontsize=18)
ax.set_ylabel('Ocean Melt', fontsize=18)
cbar=plt.colorbar()
cbar.set_label('Likelihood', fontsize=18)
ax.set_xlim([0,1.])
ax.set_ylim([0,1.])


fig, ax=plt.subplots()
plt.pcolor(np.linspace(-0.05,1.05,12), np.linspace(-0.05,1.05,12),Ls.sum(axis=0).T/np.sum(Ls))
ax.set_xlabel('Viscosity', fontsize=18)
ax.set_ylabel('Ocean Melt', fontsize=18)
cbar=plt.colorbar()
cbar.set_label('Likelihood', fontsize=18)
ax.set_xlim([0,1.])
ax.set_ylim([0,1.])


#if 1:#projecting
#    linv_A = np.linalg.solve(A.T.dot(A), A.T)
#https://stackoverflow.com/questions/2250403/left-inverse-in-numpy-or-scipy
#current thoughts: projection of Obs on PC would be perfect IF obs would be in PC (+discrepancy) space, 
#the part of obs OUTSIDE of this space has no spatial correlation treatement. i.e. not perfect but imrovement








#%%%


# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 12:11:45 2018

@author: Andreas Wernecke
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
from netCDF4 import Dataset


def setup_emulator(sim_in, sim_out, s='none', lscales=0.5):
    """Sets up an gaussian process emulator for a given set of n output values sim_out [n]
     for a set of input values sim_in [n x m] where m is the dimension of inputs.
    
    Hyperparameters are optimized at marginal likelyhood with 10 restarts of optimization.
    
    Output: A GPy Gaussian process model, the kernel/covarianze function and
    mean function.
    lscales can be given as scalar or array length dim. Default=0.5 for all
    """
    if np.shape(sim_in)[0]!=len(sim_out):
        if np.shape(sim_in)[1]==len(sim_out):
            sim_in=sim_in.T
        else: print('Wrong Shape!')
    dim=np.shape(sim_in)[1]
    if np.shape(np.shape(sim_out))[0]!=2:
        sim_out=sim_out.reshape([-1,1])
    if s=='none': s=1.
    lengthscales=np.ones(dim)*lscales
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
    gpm.optimize_restarts(10, messages=1)
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

def f_Z(Z, eta, v, ur, sr, disc, Z_err=1.): #will not work because I removed dhdt_mean as is now subfromobs
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
    
def L_RMS(red_obs, eta, sr):
    ddhdt = ur_obsgr.dot(eta*sr)
    return np.sqrt(np.mean(np.square(red_obs-ddhdt)))
    
def L_PC(eta_traget, eta, sr, eta_target_err=1., log=False):
    #eta_target contains ~1 times mean, emulators obviously not, I add it here but could be ignored as
    #constant with obs (derivation from one is caused by obs not in PC space)
    #I am quite sure error shoul not be constant
    #eta, sr = np.hstack((1.,eta)), np.hstack((1.,sr))
    if np.shape(np.shape(eta_target_err))[0]==0: eta_target_err=np.ones_like(eta)*eta_target_err
    if log:
        fs=-(eta_traget-sr*eta)**2/(2.*eta_target_err**2)
        return np.sum(fs)
    else:
        fs=np.exp(-(eta_traget-sr*eta)**2/(2.*eta_target_err**2))
        #print('FS: ')
        #print(fs)
        #print('FS*Sr: ')
        #print(fs*sr)
        #print('sr: ')
        #print(sr)
        #print('fs/fs*sr: ')
        #print(fs/(fs*sr))
        fs2=np.prod(fs*sr)#/Z_err maybe add 1/Zerr back 
        #print(fs2/np.prod(fs))
        return fs2
    
def model2obsgrid(field):
    #we are quick and dirty here!

    field=zoom(field.reshape([256,224]), 2, order=1) #this works slidly unexpected
    if 'nomodel' in globals():
        field=np.ma.array(field, mask=zoom(nomodel.reshape([256,224]), 2, order=0))
    else:
        field=np.ma.array(field)
        print('No model mask applied in model2obsgrid')
    field=field[2:, :-3].reshape([102, 5, 89, 5]).mean(axis=1).mean(axis=2)# ~500m off
    field=field[19:-11,9:-26]
    if 'noobs' in globals():
         field=np.ma.array(field, mask=np.logical_or(field.mask, noobs))
    else:
        print('No obs mask applied in model2obsgrid')
    return field
    
def calc_rec_err(z, B, disc, var_disc, var_z):
    #calculates recreation error as in https://doi.org/10.1080/01621459.2018.1514306 EQ(8)
    #Z:Observational field (flattened), so that all mean fields are substracted 
    #(as from emulator setup and eventually discrepancy)
    #B basis matrix, disc discrepancy field, 
    #var_disc: scalar variance of scale factor of disc
    #var_Z: variance field of indpendant noise (observational error) 
    #there is a lot to optimze numerically here!
    ngrid=len(z)
    covar_obs=np.identity(ngrid)
    covar_disc = np.meshgrid(disc, disc)[0] * np.meshgrid(disc, disc)[1]
    #lets better remove the disc mean from everything
    W= var_z * covar_obs + var_disc * covar_disc
    W_inv=np.linalg.inv(W)
    v=z-B.dot(np.linalg.inv(B.T.dot(W_inv).dot(B))).dot(B.T).dot(W_inv).dot(z)
    Rw=v.T.dot(W_inv).dot(v)
    return Rw

def L_3d(PC_emus, target_PC, sr, eta_target_err=1, dim=5, Cone=1, bedZero=1, size=[11,11,11], plot=False, log=False):
    #need to combine eta_target_error with PC consvar
    Ls=np.zeros(size)
    for i, v1 in enumerate(np.linspace(0,1,size[0])):
        for j, v2 in enumerate(np.linspace(0,1.,size[1])):
            for kk, v3 in enumerate(np.linspace(0.,1.,size[2])):
                #print(np.sum(predict_PC(ur, sr, PC_emus, Vs[:,i])))    
                if dim==3: V=np.array([v1,v2,v3])
                elif dim==5: V=np.array([v1,v2,v3,Cone,bedZero])
                else: print('WTF')
                PC_cons, PC_cons_var = predictXk(PC_emus, V)

                Ls[i,j,kk]=L_PC(target_PC, PC_cons, sr, eta_target_err=eta_target_err, log=log)
                #Ls[i,j,kk]=L_RMS(reduced_obs[sp_mask==0], PC_cons, sr)
                
                #print(L)
    if plot: L_3d_plot(Ls, Cone, bedZero)
    return Ls

def L_3d_plot(Ls, Cone, bedZero, Ls_total=True):
    if Cone==1:Cstr='C=1, '
    elif Cone==0: Cstr='C=1/3, '
    if bedZero==1: bedstr='Bedmap'
    elif bedZero==0: bedstr='Modified Bed'
    norm=abs(np.sum(Ls))

    title='Likelihood to be optimal; ' + Cstr + bedstr
    fig, axes=plt.subplots(nrows=2, ncols=2)
    fig.suptitle(title, fontsize=18)
    
    ax=axes[0,0]
    pcol=ax.pcolor(np.linspace(-0.05,1.05,12), np.linspace(-0.05,1.05,12),Ls.sum(axis=2).T/norm)
    #ax.set_xlabel('Traction', fontsize=18)
    ax.set_ylabel('Viscosity', fontsize=18)
    #ax.set_title(title)
    plt.colorbar(pcol, ax=ax)
    #cbar.set_label('Likelyhood', fontsize=18)
    ax.set_ylim([0,1.])
    ax.set_xlim([0,1.])
    Ls_tmp=Ls.sum(axis=2).T/norm
    imax=np.nonzero(Ls_tmp==np.max(Ls_tmp))
    ax.scatter(np.linspace(0.,1.,11)[imax[1]], np.linspace(0.,1.,11)[imax[0]], marker='*')
    #ax.set_ylim([0,1.])
    
    ax=axes[1,0]
    pcol=ax.pcolor(np.linspace(-0.05,1.05,12), np.linspace(-0.05,1.05,12),Ls.sum(axis=1).T/norm)
    ax.set_xlabel('Traction', fontsize=18)
    ax.set_ylabel('Ocean Melt', fontsize=18)
    plt.colorbar(pcol, ax=ax)
    #cbar.set_label('Likelihood', fontsize=18)
    ax.set_xlim([0,1.])
    ax.set_ylim([0,1.])
    Ls_tmp=Ls.sum(axis=1).T/norm
    imax=np.nonzero(Ls_tmp==np.max(Ls_tmp))
    ax.scatter(np.linspace(0.,1.,11)[imax[1]], np.linspace(0.,1.,11)[imax[0]], marker='*')    
    
    ax=axes[1,1]
    pcol=ax.pcolor(np.linspace(-0.05,1.05,12), np.linspace(-0.05,1.05,12),Ls.sum(axis=0).T/norm)
    ax.set_xlabel('Viscosity', fontsize=18)
    #ax.set_ylabel('Ocean Melt', fontsize=18)
    plt.colorbar(pcol, ax=ax)
    #cbar.set_label('Likelihood', fontsize=18)
    ax.set_xlim([0,1.])
    ax.set_ylim([0,1.])
    Ls_tmp=Ls.sum(axis=0).T/norm
    imax=np.nonzero(Ls_tmp==np.max(Ls_tmp))
    ax.scatter(np.linspace(0.,1.,11)[imax[1]], np.linspace(0.,1.,11)[imax[0]], marker='*')
    
    ax=axes[0,1]
    ax.plot(np.linspace(0.,1.,11), Ls.sum(axis=0).sum(axis=0)/norm, label = 'Melt')
    ax.plot(np.linspace(0.,1.,11), Ls.sum(axis=0).sum(axis=1)/norm, label = 'Viscosity')
    ax.plot(np.linspace(0.,1.,11), Ls.sum(axis=1).sum(axis=1)/norm, label = 'Traction')
    ax.legend(loc='best')
    
    if Ls_total: 
        fig.text(0.72, 0.89, 'Total L: {:.2e}'.format(np.sum(Ls)))
    if 'plotstr' in globals():
        fig.text(0.95, 0.5, plotstr, rotation='vertical', va='center')
        
def score_discr_var_func(disc_var, mean, individuals, obs_var):
    """
    Negative score function (to be minimized) to find discrepancy variance by setting it so that
    95% of (individuals-mean)^2 are within 9*(var_obs + var_disc) (i.e. 3 combined sigma)
    """    
    di2s=np.square(mean-individuals)/(disc_var+obs_var)
    score=np.sum(di2s<9)/len(di2s)
    if score>0.95: score=1.9-score
    return score

def opt_score_myself(func, args, bounds):
    """
    Optimizing (maximising) a score function (func) with a single input (x) by waves of sampling. 
    This needs the score to be monotonic everywhere but at the one local maxia == the global maxima.
    args are dditional arguments and bounds the bounds of x within which to search.
    """
    #i=9
    #func=score_discr_var_func 
    #args=(target_PC[i], vr[i,:]*sr[i],target_PC_err[i]**2)
    #bounds=[0., 750.**2]
    mean, individuals, obs_var = args #for now
        
    nwaves=6 #number of times to refocus, could be dynamic maybe
    ncandid=50 #each wave refocusses to 2/30 of the range of the last one
    x_min, x_max = bounds
    for wave in range(nwaves):
        x_candidate=np.linspace(x_min, x_max, ncandid)
        scores=np.zeros(ncandid)
        for i in range(ncandid):
            scores[i]=func(x_candidate[i], mean, individuals, obs_var)
        int_optimals=np.nonzero(scores==np.max(scores))[0]
        #for when the score flattens out, we proceed with the central end 
        if len(int_optimals)==1:
            in_optimal=int_optimals[0]
        else:
            if int_optimals[-1]==ncandid-1 & int_optimals[0]==0:
                print("ITS A PLANE MAN ITS A PLANE!!! (wave= {})".format(wave))
                in_optimal=int(np.median(int_optimals))
            elif int_optimals[-1]==ncandid-1:
                in_optimal=int_optimals[0]
            elif int_optimals[0]==0:
                in_optimal=int_optimals[-1]
            else:
                in_optimal=int(np.median(int_optimals))
        
        if in_optimal==0:
            x_min=x_candidate[in_optimal]
            x_max=x_candidate[in_optimal+1]
        elif in_optimal==ncandid-1:
            x_min=x_candidate[in_optimal-1]
            x_max=x_candidate[in_optimal]
        else:
            x_min=x_candidate[in_optimal-1]
            x_max=x_candidate[in_optimal+1]
            
        #print(scores)
    return x_candidate[in_optimal], scores[in_optimal]
    
#%%%


fakeobs=0
if fakeobs: 
    noise_f=0.5
Cone_f=1
bedZ_f=0
k=10
trainemus=0
period_07=True


var_plots=False

#%%%
#create string for remembering what we do right now
if fakeobs:
    plotstr='POC; C={}; Bed:{}; Noise={}; k={}'.format\
        (['1/3', '1'][Cone_f], ['Mod', 'BM-2'][bedZ_f], noise_f, k)
else:
    plotstr='Konrad et al.; k={}'.format(k)

print(plotstr)


datadir= "/home/dusch/Dropbox/mypaper/"
#datadir= "/home/andi/Dropbox/mypaper/"


if period_07:#7a
    dhdt=np.load(datadir+"dhdt_centered_07_v002.npy")
    dhdt_mean=np.load(datadir+"dhdt_mean_07_v002.npy")
    Vs=np.load(datadir+"Vs_07_v002.npy")
    if 0:#just a quick try:
        dhdt=dhdt[:,Vs[4,:]==0]
        Vs=Vs[:,Vs[4,:]==0]

else:#50a
    dhdt=np.load(datadir+"dhdt_centered_50_v002.npy")
    dhdt_mean=np.load(datadir+"dhdt_mean_50_v002.npy")
    Vs=np.load(datadir+"Vs_50_v002.npy")

x, y = np.load(datadir+"xy_estimate_v001.npy")

nruns=np.shape(dhdt)[1]
nomodel=np.min(dhdt, axis=1)==np.max(dhdt, axis=1)

#traction
#viscosity
#melt rate
#C==one
#Bed==bedmap

#for i in range(np.shape(dhdt)[1]):
#    print(np.sum(dhdt_mean+dhdt[:,i]))

u,s,v = np.linalg.svd(dhdt, full_matrices=False)
ur,sr,vr = u[:,:k], s[:k], v[:k,:]
#dhdt=u.dot(np.diag(s)).dot(v)
#u:PCs, s:sigular values (importance of pc); v:connets Vs to PCs
#v to be replaced by emulators
#set k=5 for now, si/sum(s)<0.05 for i>k

if trainemus:#train emulators
    PC_emus=[]
    for i in range(k):
        PC_emus.append(setup_emulator(Vs, vr[i,:], s='none'))
    if period_07:
        np.save(datadir+'emus_07.npy', PC_emus)
    else:
        np.save(datadir+'emus_50.npy', PC_emus)
elif k<=20:
    if period_07:
        PC_emus=np.load(datadir+'emus_07.npy')[:k]
    else:
        PC_emus=np.load(datadir+'emus_50.npy')[:k]

#LOO Validation
if 0:#takes about 5h :)
    #get average RMSE within decomposed, truncated ensamble as basis to judge emulator error only
    if 0:
        RMSEs=[]
        for i in range(nruns):
            for j in range(i+1, nruns):
                RMSEs.append(np.sqrt(np.mean(np.square(ur.dot(np.diag(sr)).dot(vr[:,i])-ur.dot(np.diag(sr)).dot(vr[:,j])))))
                #print(np.sqrt(np.mean(np.square(dhdt[:,i]-dhdt[:,j]))))
        mean_RMSE=np.mean(RMSEs)
        np.save(datadir+'RMSE_base.npy', mean_RMSE)
    else:
        mean_RMSE=np.load(datadir+'RMSE_base.npy')
        
    LOO_v_vec, LOO_var_v_vec = np.zeros([nruns, k]), np.zeros([nruns, k])
    for out in range(nruns):#this is the leave one out loop
        print(out)
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
        LOO_v_vec[out,:], LOO_var_v_vec[out,:] = predictXk(PC_emus_cut, Vs[:,out])
         #dhdt
        if 0:
            fig, axes = plt.subplots(1,2)
            #axes[0].pcolormesh(x, y, dhdt_cut.reshape([256,224]), vmin=-10, vmax=10)
            pcol=axes[1].pcolormesh(x, y, dhdt[:,out].reshape([256, 224]), vmin=-10, vmax=10)
            plt.colorbar(pcol, ax=axes[1])
            #axes[0].
        #LOO_mu[out], LOO_var[out] = gpm.predict(X[out,:].reshape(1,-1))
        #print "Predicted: " + str(LOO_mu[out])
        #print "Simulated: " + str(ddat[year,out])
        #print gpm.rbf.lengthscale
    np.save(datadir+'LOOv_LOOvarv.npy', [LOO_v_vec,LOO_var_v_vec])
    #for i in range(k):
    #    plt.figure()
    #    plt.scatter(
    #dhdt_cut = predict_PC(ur, sr,PC_emus_cut, Vs[:,out])
    #LOO_RMSE[out]=np.sqrt(np.mean(np.square(dhdt_cut-ur.dot(np.diag(sr)).dot(vr[:,out]))))
       

#%%%
#calibration Data
if np.shape(Vs)[0]==3:
    central_r=np.nonzero( (Vs[0]==0.5) & (Vs[1]==0.5) & (Vs[2]==0.5) )[0][0]
elif np.shape(Vs)[0]==5:
    central_r=np.nonzero( (Vs[0]==0.5) & (Vs[1]==0.5) & (Vs[2]==0.5) & (Vs[3]==Cone_f) & (Vs[4] ==bedZ_f) )[0][0]
    
if fakeobs: #fake obs from central run with noise
    if 1:
        #GP to produce "noise"
        lengthscales=np.ones(1)*100000
        gpk=GPy.kern.Matern52(input_dim=2, variance=1., lengthscale=lengthscales)
        gpm=GPy.models.GPRegression(np.array([[-1e5,-1e5]]), np.array([[0]]), gpk, noise_var=0)
        obs_noise=gpm.posterior_samples_f(np.vstack([x[::4,::4].flatten(), \
            y[::4,::4].flatten()]).T, full_cov=True, size=1)
        #obs_noise=gpm.predict(np.vstack([x.flatten(), y.flatten()]).T)
    
        obs=dhdt[:,central_r].reshape([256, 224])+noise_f*zoom(obs_noise.reshape([64, 56]), 4., order=3)
        nomodel=np.min(dhdt, axis=1)==np.max(dhdt, axis=1)
        obs=np.ma.array(obs, mask=nomodel)
        np.save(datadir+'fake_obs.npy', obs.data)
        np.save(datadir+'fake_obsm.npy', obs.mask)
    else:
        obs=np.ma.array(np.load(datadir+'fake_obs.npy'), mask=np.load(datadir+'fake_obsm.npy'))
    obs=model2obsgrid(obs+dhdt_mean.reshape([256,224]))


if not fakeobs:
    with Dataset(datadir+'dhdt_10km_sig0p5_0p0.nc') as f:
        x_obs=f.variables['x'][:]
        y_obs=f.variables['y'][:]
        year_obs=f.variables['t'][:]
        dhdt_obs=f.variables['dhdt'][:,:,:]
    dhdt_obs.mask=np.isnan(dhdt_obs.data)
    
    x_obs, y_obs = np.meshgrid(x_obs, y_obs)
    
    obs=np.mean(dhdt_obs[-14:,:,:], axis=0)
    noobs=obs.mask


#match obs (10km) and model (4km) by interpolate model to 2 and take 5X5 mean
#lets do this with the PC to speed things up

#define basemap and bring obs+model to it
m=Basemap(width=5400000., height=5400000., projection='stere',\
          ellps='WGS84', lon_0=180., lat_0=-90., lat_ts=-71., resolution='i')
                              
x_sp, y_sp = m(0, -90)

x2km=zoom(x.reshape([256,224]), 2, order=1)
y2km=zoom(y.reshape([256,224]), 2, order=1)

xbase, ybase=-1*x2km+x_sp, -1*y2km[::-1,:]+y_sp #transform

xbase=xbase[2:, :-3].reshape([102, 5, 89, 5]).mean(axis=1).mean(axis=2)#reduce res
ybase=ybase[2:, :-3].reshape([102, 5, 89, 5]).mean(axis=1).mean(axis=2)

xbase=xbase[19:-11,9:-26]#cut to area
ybase=ybase[19:-11,9:-26]#cut to area

#dhdt2km=zoom((dhdt_mean+dhdt[:,central_r]).reshape([256,224]), 2, order=1)
#dhdt2km=np.ma.array(dhdt2km, mask=dhdt2km==0)
#dhdt2km=dhdt2km[2:, :-3].reshape([102, 5, 89, 5]).mean(axis=1).mean(axis=2)
#dhdtbase=dhdt2km[19:-11,9:-26]
dhdtbase=model2obsgrid(dhdt_mean.flatten()+dhdt[:,central_r])

if fakeobs:
    x_obs, y_obs = xbase, ybase
else:
    x_obs, y_obs = -1*x_obs*1000.+x_sp, -1*y_obs*1000+y_sp
#xbase, ybase are now = x_obs, y_obs

if 1: #and some plotting
    fig, ax=plt.subplots()
    m.pcolormesh(xbase, ybase, dhdtbase, cmap='plasma', vmin=-8, vmax=0)
    cbar=plt.colorbar()
    cbar.set_label('Surface Elevation Change [m/a]', fontsize=18)
    antt.plotlines(m, ax=ax, path='/home/dusch/home_zmaw/phd/programme/surfacetypes/')
    ax.set_xlim([3.9e6, 4.4e6])
    ax.set_ylim([2.7e6, 3.4e6])

    fig, ax=plt.subplots()
    m.pcolormesh(x_obs, y_obs, obs,cmap='plasma', vmin=-8, vmax=0)    
    cbar=plt.colorbar()
    cbar.set_label('Surface Elevation Change [m/a]', fontsize=18)
    antt.plotlines(m, ax=ax, path='/home/dusch/home_zmaw/phd/programme/surfacetypes/')
    ax.set_xlim([3.9e6, 4.4e6])
    ax.set_ylim([2.7e6, 3.4e6])
    
    fig, ax=plt.subplots()
    #m.pcolormesh(x_obs, y_obs, obs-dhdtbase ,cmap='seismic', vmin=-4, vmax=4)    
    m.pcolormesh(x_obs, y_obs, obs-dhdtbase ,cmap='seismic', vmin=-4, vmax=4)   
    cbar=plt.colorbar()
    cbar.set_label('Elevation Change Difference [m/a]', fontsize=18)
    antt.plotlines(m, ax=ax, path=home+'/home_zmaw/phd/programme/surfacetypes/')    
    ax.set_xlim([3.9e6, 4.4e6])
    ax.set_ylim([2.7e6, 3.4e6])
   
    
  
#%%% prepare projecting
        
#https://stackoverflow.com/questions/2250403/left-inverse-in-numpy-or-scipy
#current thoughts: projection of Obs on PC would be perfect IF obs would be in
# PC (+discrepancy) space, 
#the part of obs OUTSIDE of this space has no spatial correlation treatement. 
#i.e. not perfect but imrovement

sp_mask = noobs.flatten() #spatial mask in 1d
ur_obsgr=np.zeros([3888-np.sum(sp_mask), k])

#ur_obsgr[:,0]=model2obsgrid(dhdt_mean).flatten()
#ur_obsgr[sp_mask,0]=0.

for i in range(k):
    ur_obsgr[:,i]=model2obsgrid(ur[:,i]).flatten()[sp_mask==0]
#add discrepancy to ur_obsgr here
linv_A = np.linalg.solve(ur_obsgr.T.dot(ur_obsgr), ur_obsgr.T)



#%%%    do reprojection
    
#preparing obs (removing mean fields and fill for calib)
reduced_obs=obs.copy()#this is mostly for later but also if fakeobs
reduced_obs = reduced_obs.flatten() - model2obsgrid(dhdt_mean).flatten()


if not fakeobs: #finding error of obs PCs by comparing all times
    target_PCs=[]
    nobs=np.shape(dhdt_obs)[0]
    for i in range(nobs-14, nobs):
        obs_tmp=dhdt_obs[i,:,:].flatten()-model2obsgrid(dhdt_mean).flatten()
        target_PCs.append(linv_A.dot(obs_tmp[sp_mask==0]))
    target_PCs=np.asarray(target_PCs)
    target_PC=target_PCs.mean(axis=0)
    target_PC_err=target_PCs.std(axis=0)
    if 1:#plot PC of mean
        plt.figure()
        plt.hist(target_PCs[:,0])
else:
    target_PC = linv_A.dot(reduced_obs[sp_mask==0])
    target_PC_err=10. #you know...
    
    
    
#%%%  model discrepancy
"""
if 1:
    #first try: min distance between (dim reduced) ensemble members and obs where obs are not "sourounded" 

    dhdt_dimred=ur_obsgr.dot(np.diag(sr)).dot(vr)
    dist=dhdt_dimred.T-(obs-model2obsgrid(dhdt_mean)).flatten()
    dist=dist.T
    obs_notin_ensemble=np.logical_or(np.sum(np.sign(dist), axis=1)==np.shape(dist)[1], \
                                     np.sum(np.sign(dist), axis=1)==-1*np.shape(dist)[1])
    disc=np.zeros(np.shape(dist)[0])
    disc=np.where(np.sum(np.sign(dist), axis=1)==np.shape(dist)[1], np.min(dist,axis=1), disc)
    disc=np.where(np.sum(np.sign(dist), axis=1)==-1*np.shape(dist)[1], np.max(dist,axis=1), disc)
    disc=np.ma.array(disc, mask=obs.mask)
    var_disc=0.5#....
    #print(disc)
else:
    disc=np.zeros(np.shape(obs)[0])
    var_disc=0.
"""   
#now replaced by discrepancy in PC space
#di2=np.square((target_PC[i]-vr[i,:]*sr[i])
var_discs=np.zeros(k)
if 0:
    for i in range(k):
        var_disc_tmp, score = opt_score_myself(score_discr_var_func, args=(target_PC[i], vr[i,:]*sr[i], \
                target_PC_err[i]**2), bounds=[0., 750.**2])
        var_discs[i]=var_disc_tmp
        if score<0.9: print('Unsuccessful optimisation!!')#check for success as it should be near 0.95
    
    sig_tot=np.sqrt(var_discs+target_PC_err**2)
    
    if 1:#plot
        fig, axes = plt.subplots(2,3, figsize=(12,5))
        axes=axes.flatten()
        for i in range(6):
            axes[i].plot([target_PC[i],target_PC[i]], [0,20], c='b')
            axes[i].plot([target_PC[i]-3.*sig_tot[i], target_PC[i]-3*sig_tot[i]], [0,10], c='gray')
            axes[i].plot([target_PC[i]+3.*sig_tot[i], target_PC[i]+3*sig_tot[i]], [0,10], c='gray')
                
            axes[i].hist(vr[i,:]*sr[i], label=str(i), color='orange')
            axes[i].legend()
else:
    for i in range(k):
        var_disc_tmp, score = opt_score_myself(score_discr_var_func, args=(np.mean(vr[i,:]*sr[i]), \
            target_PCs[:,i], target_PC_err[i]**2), bounds=[0., 750.**2])
        var_discs[i]=var_disc_tmp
        if score<0.9: print('Unsuccessful optimisation!!')#check for success as it should be near 0.95
    
    sig_tot=np.sqrt(var_discs+target_PC_err**2)
    
    if 1:#plot
        fig, axes = plt.subplots(2,3, figsize=(12,5))
        axes=axes.flatten()
        for i in range(6):
            axes[i].plot([target_PC[i],target_PC[i]], [0,20], c='b')
            axes[i].plot([target_PC[i]-3.*sig_tot[i], target_PC[i]-3*sig_tot[i]], [0,10], c='gray')
            axes[i].plot([target_PC[i]+3.*sig_tot[i], target_PC[i]+3*sig_tot[i]], [0,10], c='gray')
            
            axes[i].hist(vr[i,:]*sr[i], label=str(i), color='orange')
            axes[i].legend()
            
#This blasts up uncerteinties even if the obs are in the center of wide model ensembles.
#maybe something like: choosinf sig_disc so that at least 1 ensemble member is within d<1 or d<3???
#or at lest, if like this d<5 or something, so that some constrain remains.
    
if 1:#Reconstruction error
    #var_obs=np.ones_like(reduced_obs)*5.
    #field_disc=np.random.rand(ngrid)
   
    #print(calc_rec_err(reduced_obs, ur_obsgr, disc=disc, var_disc=var_disc, var_z=var_obs))
    
    if 1:
        #fraction of variance covered in PCs
        fig, ax=plt.subplots()
        m.pcolormesh(x_obs, y_obs, obs, cmap='plasma', vmin=-5, vmax=3)
        cbar=plt.colorbar()
        cbar.set_label('Obs', fontsize=18)
        antt.plotlines(m, ax=ax, path='/home/dusch/home_zmaw/phd/programme/surfacetypes/')
        ax.set_xlim([3.9e6, 4.4e6])
        ax.set_ylim([2.7e6, 3.4e6])
        
        fig, ax=plt.subplots()
        m.pcolormesh(x_obs, y_obs, model2obsgrid(ur.dot(target_PC)+dhdt_mean)\
                     , cmap='plasma', vmin=-5, vmax=3)
        cbar=plt.colorbar()
        cbar.set_label('Reconstructed Obs', fontsize=18)
        antt.plotlines(m, ax=ax, path='/home/dusch/home_zmaw/phd/programme/surfacetypes/')
        ax.set_xlim([3.9e6, 4.4e6])
        ax.set_ylim([2.7e6, 3.4e6])
        
        fig, ax=plt.subplots()
        m.pcolormesh(x_obs, y_obs, model2obsgrid(ur.dot(target_PC)+dhdt_mean) \
                                          -obs , cmap='seismic', vmin=-3, vmax=3)
        cbar=plt.colorbar()
        cbar.set_label('Reconstruction Difference', fontsize=18)
        antt.plotlines(m, ax=ax, path='/home/dusch/home_zmaw/phd/programme/surfacetypes/')
        ax.set_xlim([3.9e6, 4.4e6])
        ax.set_ylim([2.7e6, 3.4e6])
        
    
    var_tot=np.var(reduced_obs)
    var_lost= np.var(ur_obsgr.dot(target_PC)-reduced_obs[sp_mask==0])
    print('Var_lost/var_tot = {}'.format(var_lost/var_tot))
    
    if 1: #plot lost var as func of k
        for kk in range(k):
            linv_A_tmp = np.linalg.solve(ur_obsgr[:,:kk].T.dot(ur_obsgr[:,:kk]), ur_obsgr[:,:kk].T)
            target_PCs_tmp=[]
            nobs=np.shape(dhdt_obs)[0]
            for i in range(nobs-14, nobs):
                obs_tmp=dhdt_obs[i,:,:].flatten()-model2obsgrid(dhdt_mean).flatten()
                target_PCs_tmp.append(linv_A_tmp.dot(obs_tmp[sp_mask==0]))
            target_PCs_tmp=np.asarray(target_PCs_tmp)
            target_PC_tmp=target_PCs_tmp.mean(axis=0)
            var_lost_tmp = np.var(ur_obsgr[:,:kk].dot(target_PC_tmp)-reduced_obs[sp_mask==0])
            print('Var_lost/var_tot for k{} = {}'.format(kk, var_lost_tmp/var_tot))
            
#%%%   Calibration
if 1:
    log=False
    print('Calculating likelihoods part 1/4')
    Ls1_log=L_3d(PC_emus, target_PC, sr, Cone=1, bedZero=1, eta_target_err=sig_tot, plot=True, log=log)
    
    Ls1_kart=L_3d(PC_emus, target_PC, sr, Cone=1, bedZero=1, eta_target_err=sig_tot, plot=True, log=False)
    L_3d_plot(Ls1_kart, Cone=1, bedZero=1)
    L_3d_plot(np.exp(Ls1_log), Cone=1, bedZero=1)
    
    print('Calculating likelihoods part 2/4')
    Ls2=L_3d(PC_emus, target_PC, sr, Cone=1, bedZero=0, eta_target_err=sig_tot, plot=True, log=log)
    #L_3d_plot(Ls2, Cone=1, bedZero=0)
    print('Calculating likelihoods part 3/4')
    Ls3=L_3d(PC_emus, target_PC, sr, Cone=0, bedZero=1, eta_target_err=sig_tot, plot=True, log=log)
    #L_3d_plot(Ls3, Cone=0, bedZero=1)
    print('Calculating likelihoods part 4/4')
    Ls4=L_3d(PC_emus, target_PC, sr, Cone=0, bedZero=0, eta_target_err=sig_tot, plot=True, log=log)
    #L_3d_plot(Ls4, Cone=0, bedZero=0)
    
    #for i in range(np.shape(Vs)[1]): 
    #    PC_cons, PC_cons_var = predictXk(PC_emus, Vs[:,i])
    #    L=L_PC(target_PC, PC_cons, sr, eta_target_err=10.)
    #    print('Run# '+str(i)+': '+str(L))
    
#Have to Ls to get the normalization right

#%%%












    
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
#    geofield=griddata(np.array([geox, geoy]).T, geoha, np.array([csx2d.flatten(), 
#csy2d.flatten()]).T, method='nearest').reshape(sum(m_csy), sum(m_csx))
#    csfb=cs_tmp-geofield
#    csth=csfb*1029./(1029.-916.)
#    #imcs=m.imshow(cs[::-1][, ax=ax, vmin=-40, vmax=40,cmap='bwr')#vmin, vmax=vmax)
#    imcs=m.pcolormesh(csx2d, csy2d, csth, vmin=vmin, vmax=vmax)
#    #m.pcolormesh(csx, csy, cs[::-1], vmin=vmin, vmax=vmax) #slower, but shows the same, 
#so csx/csy (or csxy[::-1]) can be used to connect with OIB data
#    plotlines(m, ax, path='/home/dusch/home_zmaw/phd/programme/surfacetypes/')
#    plt.colorbar(imcs, ax=ax)






if var_plots:    
    s2=s**2
    fig, ax=plt.subplots(1)
    ax.bar(np.arange(1,len(s2)+1),s2/np.sum(s2), color='gray', width=1.)
    ax.set_xlabel('# PC', fontsize=18)
    ax.set_ylabel('Fraction of Total Variance', fontsize=18)
    ax.set_xlim([1,20])
    ax.set_xticks(np.arange(1,21)+0.5)
    ax.set_xticklabels(np.arange(1,21))
    
    ax_c=ax.twinx()
    ax_c.set_xlim([1,22])
    cums=np.array([np.sum(s2[:i]/np.sum(s2)) for i in range(len(s2))])
    ax_c.step(np.arange(1,len(s2)+1),cums, color='r',linewidth=2)
    ax_c.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax_c.set_yticklabels(['0','0.20', '0,40', '0,60', '0.80', '1.00'])
    ax_c.set_ylim([0,1])
    #ax_c.yaxis.label.set_color('g')
    ax_c.tick_params(axis='y', colors='r')
    ax_c.set_ylabel('Commutative Fraction of Variance', fontsize=18, color='r')
    #ax_c.plot([1.5,4.5], [0.6, 0.6], linestyle='--', c='r')
    #ax_c.plot([3.5,7.5], [0.75, 0.75], linestyle='--', c='r')
    #ax_c.text(5, 0.6, '60%', color='r')
    #ax_c.text(8, 0.75, '75%', color='r')
    s2=s**2
    fig, ax=plt.subplots(1)
    ax.bar(np.arange(1,len(s2)+1),s2/np.sum(s2), color='gray', width=1.)
    ax.set_xlabel('# PC', fontsize=18)
    ax.set_ylabel('Fraction of Total Variance', fontsize=18)
    ax.set_xlim([1,20])
    ax.set_xticks(np.arange(1,21)+0.5)
    ax.set_xticklabels(np.arange(1,21))
    
    ax_c=ax.twinx()
    ax_c.set_xlim([1,22])
    cums=np.array([np.sum(s2[:i]/np.sum(s2)) for i in range(len(s2))])
    ax_c.step(np.arange(1,len(s2)+1),cums, color='r',linewidth=2)
    ax_c.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax_c.set_yticklabels(['0','0.20', '0,40', '0,60', '0.80', '1.00'])
    ax_c.set_ylim([0,1])
    #ax_c.yaxis.label.set_color('g')
    ax_c.tick_params(axis='y', colors='r')
    ax_c.set_ylabel('Commutative Fraction of Variance', fontsize=18, color='r')
    #ax_c.plot([1.5,4.5], [0.6, 0.6], linestyle='--', c='r')
    #ax_c.plot([3.5,7.5], [0.75, 0.75], linestyle='--', c='r')
    #ax_c.text(5, 0.6, '60%', color='r')
    #ax_c.text(8, 0.75, '75%', color='r')














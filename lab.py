"""Finite pixels + Gaussian diffraction + shutter timing with motion nuisance parameters."""
from pathlib import Path
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr
from scipy.optimize import least_squares

PIXELS=np.arange(-8.,9.)
TIMES=np.repeat([-1.,0.,1.],2)
DURATIONS=np.tile([.4,1.2],3)
TRUTH=np.array([8.,.6,.04,.23,.8]) # N, alpha, delta s, c px, v px/s
CAL_SD=np.array([.01,.02]) # independently measured c and v
READ_SD=2.

def sigma_px(N):
    """550 nm wavelength and 4 micrometre pitch; N sets Gaussian pupil size."""
    return .550*N/(np.sqrt(2)*np.pi*4.)

def image(theta,a=.35,times=TIMES,durations=DURATIONS,nodes=32):
    N,alpha,delta,c,v=theta
    if N<=0 or abs(alpha)>1 or np.any(np.asarray(durations)<0):
        raise ValueError('Invalid width, shutter asymmetry or duration')
    # Map quadrature from [-1, 1] to normalized exposure [-1/2, 1/2].
    z,w=leggauss(nodes); u=z/2; w=w/2
    t=np.asarray(times)[:,None]+delta+np.asarray(durations)[:,None]*u
    center=c+v*t+.5*a*t*t
    # CDF differences integrate a normalized diffraction PSF over each pixel.
    hi=(PIXELS[None,:,None]+.5-center[:,None,:])/sigma_px(N)
    lo=(PIXELS[None,:,None]-.5-center[:,None,:])/sigma_px(N)
    return 20+10000*((ndtr(hi)-ndtr(lo))@(w*(1+2*alpha*u)))

def gauge(theta,h,a):
    """Change time origin while preserving every camera prediction exactly."""
    N,alpha,delta,c,v=theta
    return np.array([N,alpha,delta+h,c-v*h+.5*a*h*h,v-a*h])

def jacobian(theta,a):
    steps=np.array([1e-4,1e-5,1e-5,1e-5,1e-5])
    return np.column_stack([((image(theta+np.eye(5)[i]*h,a)-image(theta-np.eye(5)[i]*h,a))/(2*h*READ_SD)).ravel() for i,h in enumerate(steps)])

def fit(data,a,calibration=TRUTH[3:],cal_sd=CAL_SD):
    def residual(x):
        r=((image(x,a)-data)/READ_SD).ravel()
        # Independent external-clock motion observations break the time gauge.
        return np.r_[r,(x[3:]-calibration)/cal_sd]
    r=least_squares(residual,[7.8,.5,.02,.22,.78],bounds=([3,-.95,-.5,-1,.1],[16,.95,.5,1,1.5]),xtol=1e-11,ftol=1e-11,gtol=1e-11)
    if not r.success: raise RuntimeError(r.message)
    return r.x,r

def rmse(predicted, reference):
    """Root mean square, retaining opposite-sign errors before averaging."""
    return float(np.sqrt(np.mean((np.asarray(predicted)-reference)**2)))

def demo(out='outputs',trials=120):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out); out.mkdir(exist_ok=True)
    rng=np.random.default_rng(20260927)
    report={'truth':TRUTH.tolist(),'parameter_order':['N','alpha','delta_s','c_px','v_px_per_s'],'trials':trials,'scenarios':{}}
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for a,label in [(0.,'constant'),(.35,'accelerating')]:
        clean=image(TRUTH,a); x,r=fit(clean,a)
        cov=np.linalg.inv(r.jac.T@r.jac); sd=np.sqrt(np.diag(cov))
        samples=np.array([fit(clean+rng.normal(0,READ_SD,clean.shape),a,TRUTH[3:]+rng.normal(0,CAL_SD))[0] for _ in range(trials)])
        h=np.linspace(-.1,.1,51)
        errors=[np.max(abs(image(gauge(TRUTH,q,a),a)-clean)) for q in h]
        sweep=[]
        for b in [-.04,-.02,0.,.02,.04]:
            biased,_=fit(clean,a,TRUTH[3:]+[b,0])
            sweep.append({'origin_bias_px':b,'fit':biased.tolist()})
        J=jacobian(TRUTH,a); s=np.linalg.svd(J,compute_uv=False)
        # Along the exact gauge, d(c,v)/d(delta)=(-v,-a).
        ideal_timing_floor=1/np.sqrt((TRUTH[4]/CAL_SD[0])**2+(a/CAL_SD[1])**2)
        report['scenarios'][label]={'recovered':x.tolist(),'image_singular_values':s.tolist(),'max_gauge_image_error_ADU':max(errors),'predicted_sd':sd.tolist(),'empirical_sd':samples.std(axis=0,ddof=1).tolist(),'coverage_95':np.mean(abs(samples-TRUTH)<1.96*sd,axis=0).tolist(),'perfect_image_timing_floor_s':ideal_timing_floor,'origin_bias_sweep':sweep,'heldout_rmse_ADU':rmse(image(x,a,np.array([-.5,.5]),np.array([.7,1.5])), image(TRUTH,a,np.array([-.5,.5]),np.array([.7,1.5])))}
        axs[0].plot(h*1000,[gauge(TRUTH,q,a)[3] for q in h],label=label)
        axs[1].plot([q['origin_bias_px'] for q in sweep],[(q['fit'][2]-.04)*1000 for q in sweep],'.-',label=label)
        axs[2].plot([0,1],[sd[2]*1000,samples[:,2].std(ddof=1)*1000],'.-',label=label)
    axs[0].set(xlabel='Timing gauge shift (ms)',ylabel='Equivalent trajectory origin (px)',title='Different parameters, identical images')
    axs[1].set(xlabel='Origin calibration bias (px)',ylabel='Timing estimate bias (ms)',title='Calibration bias becomes timing bias')
    axs[2].set(xticks=[0,1],xticklabels=['Predicted','Monte Carlo'],ylabel='Timing standard deviation (ms)',title='Independent motion calibration')
    for ax in axs: ax.grid(alpha=.25); ax.legend()
    fig.savefig(out/'experiment.png',dpi=160); plt.close(fig)
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': demo()

import unittest
import numpy as np
from scipy.integrate import quad
from lab import *

class PhysicsTests(unittest.TestCase):
    def test_rmse_does_not_cancel_signed_errors(self):
        self.assertEqual(rmse(np.array([-3.,3.]),np.zeros(2)),3.)

    def test_independent_space_time_integration(self):
        N,alpha,d,c,v=TRUTH; a=.35; T=.9; t=.3; j=1
        s=sigma_px(N)
        def integrand(u):
            q=t+d+T*u; center=c+v*q+.5*a*q*q
            return (1+2*alpha*u)*quad(lambda x: np.exp(-.5*((x-center)/s)**2)/(np.sqrt(2*np.pi)*s),j-.5,j+.5,epsabs=1e-11)[0]
        expected=20+10000*quad(integrand,-.5,.5,epsabs=1e-11)[0]
        self.assertAlmostEqual(image(TRUTH,a,[t],[T])[0,j+8],expected,places=8)
        # Fourier transform of exp(-r²/(2s_pupil²)) gives intensity sigma.
        pupil_s=16000/(2*N); k=2*np.pi/(.550*16000)
        amp=lambda x: quad(lambda r: np.exp(-r*r/(2*pupil_s**2))*np.cos(k*x*r),-10*pupil_s,10*pupil_s)[0]
        self.assertAlmostEqual((amp(s*4)/amp(0))**2,np.exp(-.5),places=9)

    def test_flux_and_quadrature(self):
        np.testing.assert_allclose((image(TRUTH)-20).sum(axis=1),10000,atol=1e-8)
        np.testing.assert_allclose(image(TRUTH,nodes=32),image(TRUTH,nodes=64),atol=1e-9)
        theta=TRUTH.copy(); theta[4]=0
        np.testing.assert_allclose(image(theta,0),image(theta,0,durations=np.zeros(6)),atol=1e-9)

    def test_exact_gauge_and_rank(self):
        for a in [0,.35]:
            for h in [-.1,.07]: np.testing.assert_allclose(image(TRUTH,a),image(gauge(TRUTH,h,a),a),atol=1e-9)
            J=jacobian(TRUTH,a); s=np.linalg.svd(J,compute_uv=False)
            self.assertLess(s[-1]/s[0],1e-8); self.assertGreater(s[-2]/s[0],1e-4)
            np.testing.assert_allclose(J@np.array([0,0,1,-.8,-a]),0,atol=1e-5)

    def test_recovery_and_holdout(self):
        for a in [0,.35]:
            x,_=fit(image(TRUTH,a),a)
            np.testing.assert_allclose(x,TRUTH,atol=1e-8)
            np.testing.assert_allclose(image(x,a,[.4],[1.5]),image(TRUTH,a,[.4],[1.5]),atol=1e-7)

    def test_calibration_bias(self):
        x,_=fit(image(TRUTH,0),0,TRUTH[3:]+[.04,0])
        self.assertAlmostEqual(x[2]-.04,-.04/.8,places=7)
        np.testing.assert_allclose(image(x,0),image(TRUTH,0),atol=1e-7)

    def test_covariance(self):
        rng=np.random.default_rng(714)
        for a in [0,.35]:
            clean=image(TRUTH,a); _,r=fit(clean,a)
            sd=np.sqrt(np.diag(np.linalg.inv(r.jac.T@r.jac)))
            samples=np.array([fit(clean+rng.normal(0,2,clean.shape),a,TRUTH[3:]+rng.normal(0,CAL_SD))[0] for _ in range(100)])
            ratio=samples.std(axis=0,ddof=1)/sd
            self.assertTrue(np.all((ratio>.7)&(ratio<1.35)),ratio)
            self.assertTrue(np.all(abs(samples.mean(axis=0)-TRUTH)<.4*sd))

if __name__=='__main__': unittest.main()

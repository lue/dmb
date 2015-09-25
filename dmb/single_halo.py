import numpy as np
from colossus.Cosmology import *
from colossus.HaloConcentration import *
from colossus.HaloDensityProfile import *
from scipy.stats import norm

def NFWa(r,rs,rho0,alpha=1):
    '''
    Modified NFW profile (double power law). Used to describe profiles in the smallest halos (Ishiyama 2014)
    :param r:
    :param rs:
    :param rho0:
    :param alpha:
    :return: density profile
    '''
    return rho0 * (r/rs)**-alpha * (1.0+r/rs)**(alpha-3.0)

def IshiyamaAlpha(M):
    '''
    Fitting function for alpha (parameter for modified NFW) from Ishiyama 2014:
    http://adsabs.harvard.edu/abs/2014ApJ...788...27I
    :param M: Virial mass of a halo in solar masses.
    :return: alpha (inner slope)
    '''
    alpha = -0.123*np.log10(M/1e-6)+1.461
    if isinstance(alpha, float):
        alpha = np.max([1.0,   alpha])
        alpha = np.min([1.461, alpha])
    else:
        alpha[alpha < 1] = 1.
        alpha[alpha > 1.461] = 1.461
    return alpha

def HaloBoost(z, M, c, alpha):
    '''
    Boosting factor of an individual halo
    :param z: redshift of the halo
    :param M: mass of the halo (in Msun)
    :param c: concentration
    :param alpha: inner slope
    :return: Boosting factor of an individual halo
    '''
    profile = NFWProfile(M=M, mdef='vir', z=z, c=c)
    rho0, rs = profile.fundamentalParameters(M, c, z, 'vir')
    Rmax= c*rs
    R = np.logspace(np.log10(Rmax)-24, np.log10(Rmax), 1000)
    rho = NFWa(R, rs, rho0, alpha=alpha)
    V = np.concatenate([[0], 4./3.*np.pi*R**3])
    V = np.diff(V)
    rho2V = rho**2*V
    B_nu = rho2V.sum() * V.sum() / M**2
    return B_nu

def HaloBoost_c_backup(z, M, c, alpha, cs):
    '''
    Boosting factor of a group of halos with given mass and a scatter in concentrations
    :param z: redshift of the halo
    :param M: mass of the halo (in Msun)
    :param c: concentration
    :param cs: concentration spread (in log10)
    :param alpha: inner slope
    :return:
    '''
    rand_c = 10**np.random.normal(np.log10(c), cs, 1000)
    temp = 0
    for i in range(len(rand_c)):
        temp += HaloBoost(z, M, rand_c[i], alpha)
    return temp / len(rand_c)


def HaloBoost_c(z, M, c, alpha, cs):
    '''
    Boosting factor of a group of halos with given mass and a scatter in concentrations
    :param z: redshift of the halo
    :param M: mass of the halo (in Msun)
    :param c: concentration
    :param cs: concentration spread (in log10)
    :param alpha: inner slope
    :return:
    '''
    x = np.linspace(norm.ppf(0.001), norm.ppf(0.999), 12)
    y = norm.pdf(x)
    y /= y.sum()
    rand_c = 10**(np.log10(c) + cs * x)
    temp = 0
    for i in range(len(rand_c)):
        temp += HaloBoost(z, M, rand_c[i], alpha) * y[i]
    return temp
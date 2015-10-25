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

def IshiyamaAlpha(M, Mcrit):
    '''
    Fitting function for alpha (parameter for modified NFW) from Ishiyama 2014:
    http://adsabs.harvard.edu/abs/2014ApJ...788...27I
    :param M: Virial mass of a halo in solar masses.
    :return: alpha (inner slope)
    '''
    alpha = -0.123*np.log10(M/Mcrit)+1.461
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
    if alpha==1:
        R = np.logspace(np.log10(rs)-2, np.log10(Rmax), 100)
    else:
        R = np.logspace(np.log10(Rmax)-24, np.log10(Rmax), 1000)
    rho = NFWa(R, rs, rho0, alpha=alpha)
    V = np.concatenate([[0], 4./3.*np.pi*R**3])
    V = np.diff(V)
    rho2V = rho**2*V
    B_nu = rho2V.sum() * V.sum() / M**2
    return B_nu, V.sum(), c, rs

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
    res = 0
    for i in range(len(rand_c)):
        temp, Vsum, c, rs = HaloBoost(z, M, rand_c[i], alpha)
        res += temp * y[i]
    return res, Vsum, c, rs

def subhaloMF(M,A,zeta,m):
    '''
    Simple subhalo mass function model
    http://arxiv.org/pdf/1312.1729v3.pdf Eq.2
    :param M: mass of the host halo
    :param A: normalizing constant
    :param zeta: power law coefficient
    :param m: mass of subhalos
    :return: dn/dm
    '''
    return A/M*(m/M)**-zeta

def HaloBoost_sub(z, M, cs, Mmin, A, zeta, Ishiyama):
    c = concentration(M, 'vir', z, model='diemer15')
    alpha = IshiyamaAlpha(M, Mmin)
    if not Ishiyama:
        alpha = 1.0
    BM = HaloBoost_c(z, M, c, alpha, cs)
    # m_list = np.logspace(np.log10(Mmin), np.log10(M), 100)
    m_list = 10**np.arange(np.log10(Mmin), np.log10(M)+0.001, 0.1)
    dndm = subhaloMF(M, A, zeta, m_list)
    c_m = concentration(m_list, 'vir', z, model='diemer15')
    alpha_m = IshiyamaAlpha(m_list, Mmin)
    if not Ishiyama:
        alpha_m = alpha_m*0+1.0
    Bm = np.zeros(len(m_list))
    Vsum = np.zeros(len(m_list))
    rs_m = np.zeros(len(m_list))
    for i in range(len(Bm)-1):
        Bm[i], Vsum[i], c_m[i], rs_m[i] = HaloBoost_c(z, m_list[i], c_m[i], alpha_m[i], cs)
    Mh = 1.0*M
    Mh -= ((m_list[1:]+m_list[:-1])/2.0*(dndm[1:]+dndm[:-1])/2.0*np.diff(m_list)).sum()
#     print (Bm / Vsum * m_list**2)
    R = np.sum((Bm[:-1] / Vsum[:-1] * m_list[:-1]**2 * (dndm[1:]+dndm[:-1])/2.0*np.diff(m_list))) + (BM[0] / BM[1] * M**2) * Mh/M
    return BM, R * BM[1] / M**2

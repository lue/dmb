import numpy as np
from colossus.Cosmology import *
from colossus.HaloConcentration import *
from colossus.HaloDensityProfile import *

def myPS(M, z, cosmo):
    '''
    My implementation of PS mass function
    :param M: mass
    :param z: redshift
    :param cosmo: cosmology
    :return:
    '''
    deltac = Cosmology.AST_delta_collapse
    R = cosmo.lagrangianR(M)
    sigma = cosmo.sigma(R, j=0, z=z)
    Rup = cosmo.lagrangianR(M*1.01)
    Rdw = cosmo.lagrangianR(M/1.01)
    dsigmadM = - (cosmo.sigma(Rup, j=0, z=z) - cosmo.sigma(Rdw, j=0, z=z)) / (0.02 * M)
    dndM = np.sqrt(2/np.pi) * np.exp(-deltac**2 / 2 / sigma**2) * deltac / sigma**2 * dsigmadM / M
    return dndM * cosmo.matterDensity(0.0)

def SMT(M, z, cosmo):
    A = 0.3222
    a = 0.707
    p = 0.3
    deltac = Cosmology.AST_delta_collapse
    R = cosmo.lagrangianR(M)
    sigma = cosmo.sigma(R, j=0, z=z)
    Rup = cosmo.lagrangianR(M*1.01)
    Rdw = cosmo.lagrangianR(M/1.01)
    dsigmadM = - (cosmo.sigma(Rup, j=0, z=z) - cosmo.sigma(Rdw, j=0, z=z)) / (0.02 * M)
    dndM = A*np.sqrt(2*a/np.pi) *(1.0 + (sigma**2/a/deltac**2)**p) * deltac / sigma**2 * np.exp(-a*deltac**2 / 2 / sigma**2) * dsigmadM / M
    return dndM * cosmo.matterDensity(0.0)

def SHMF(M, Mcrit):
    '''
    Subhalo Mass Function in a form presented in:
    http://arxiv.org/pdf/1403.6827v1.pdf
    and
    http://arxiv.org/pdf/1403.6835v1.pdf (Eq. 6)
    :param M:
    :return:
    '''
    Am = 0.09
    alpha = 0.82
    m = np.logspace(np.log10(Mcrit), np.log10(M), 100)
    mM = m/M
    dNdlogmM = Am*mM**(-alpha)*np.exp(-50.0*mM**4)
    dNdmM = dNdlogmM / M
    return m, dNdmM, dNdlogmM

# m, dNdmM, dNdlogmM = SHMF(1e-3,1e-3)
# plt.plot(m_sh, dNdmM); plt.xscale('log'); plt.yscale('log')

def ConditionalMF(M, z, Rsk, cosmo):
    '''
    :param M: Mass of the host halo
    :param z: Redshift
    :return: Conditional mass function (Eq. 15 Shneider 2014)
    '''
    m_list = np.logspace(-9, np.log10(M), 100)
    r_list = cosmo.lagrangianR(m_list)
    sigma_list = cosmo.sigma(r_list,j=0.0,z=z)
    return m_list, 1.0/44.5/(6.0*np.pi**2)*(M/m_list)*cosmo.matterPowerSpectrum(1.0/Rsk)/ \
           (Rsk**3*np.sqrt(2.0*np.pi*(sigma_list**2 - cosmo.sigma(cosmo.lagrangianR(M),j=0.0,z=z)**2)))


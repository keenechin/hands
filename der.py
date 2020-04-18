import numpy as np

def kappa(phi):
    #phi is the angle between the vector between node n-1 and n
    # and the vector between node n and n+1
    #kappa is curvature and goes from -inf to inf
    kappa = 2*np.tan(phi/2)
    return kappa
def phi(kappa):
    phi = 2 * np.arctan2(kappa,2)
    return phi

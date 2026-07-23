
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

def validate_interpolation_data(x, y):
    """Validate data for interpolation"""
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length. Got {len(x)} and {len(y)}")

    if len(x) < 2:
        raise ValueError(f"Need at least 2 points for interpolation. Got {len(x)} points")

    if not np.all(np.isfinite(x)) :
        raise ValueError("x contains non-finite values (NaN or Inf)")
    if not np.all(np.isfinite(y)):
        for v in y:
            print(v)
        raise ValueError("y contains non-finite values (NaN or Inf)")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x values must be strictly increasing")

def Legendre_coefficients_v1(mu,phase_function,g,twoM,x,w,Pl ,smooth_phase=False):

    idx = np.argsort(mu)
    mu_values = mu[idx]
    phase_values = phase_function[idx]
    if smooth_phase:
        window_size = 15
        phase_values = moving_average(phase_values, window_size)
    try:
        validate_interpolation_data(mu_values, phase_values)
        int_plt = PchipInterpolator(mu_values, phase_values)
        f11 = int_plt(x)

    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        print(f"Data diagnostics:")
        print(
            f"  mu_values: shape={mu_values.shape}, has_nan={np.any(np.isnan(mu_values))}, has_inf={np.any(np.isinf(mu_values))}")
        print(
            f"  phase_values: shape={phase_values.shape}, has_nan={np.any(np.isnan(phase_values))}, has_inf={np.any(np.isinf(phase_values))}")
        raise


    S0 = sum(f11 * w) * 0.5
    G0 = sum(f11 * x * w) * 0.5
    a = g / G0
    b = 1 - a * S0
    Px = a * f11 + b
    gl = 0.5 * np.sum(Px[:,np.newaxis] * Pl * w[:,np.newaxis],axis=0)
    alpha = gl * (2 * np.arange(0, twoM) + 1.0)
    return gl,alpha

def Legendre_coefficients(mu,phase_function,twoM):
    x, w = gauss_zeroes_weights(-1, 1, twoM)
    idx = np.argsort(mu)
    pchip = PchipInterpolator(mu[idx], phase_function[idx])
    f11 = pchip(x)
    norm_factor = 0.5 * np.sum(f11*w)
    f11 /= norm_factor
    Pl = legendre_polynomial(x, twoM)
    gl = 0.5 * np.sum(f11[:,np.newaxis] * Pl * w[:,np.newaxis],axis=0)
    alpha = gl * (2 * np.arange(0, twoM) + 1.0)
    return gl,alpha,Pl,f11,x,w


def legendre_polynomial(x, kmax):
    nk = kmax
    xx = np.array(x)
    nx = xx.size
    pk = np.zeros((nx,nk))
    if kmax == 0:
        pk[:,0] = 1.0
    elif kmax == 1:
        pk[:,0] = 1.0
        pk[:,1] = x
    else:
        pk[:,0] = 1.0
        pk[:,1] = x
        for ix in range(nx):
            for ik in range(2, nk):
                pk[ix,ik] = (2.0 - 1.0 / ik) * xx[ix] * pk[ix,ik - 1] - (1.0 - 1.0 / ik) * pk[ix,ik - 2]

    return pk

def schmidt_polynomial(m, x, kmax):
    nk = kmax + 1
    qk = np.zeros(nk)
    #
    #   k=m: Qmm(x)=c0*[sqrt(1-x2)]^m
    c0 = 1.0
    for ik in range(2, 2 * m + 1, 2):
        c0 = c0 - c0 / ik
    qk[m] = np.sqrt(c0) * np.power(np.sqrt(1.0 - x * x), m)
    #
    #	Q{k-1}m(x), Q{k-2}m(x) -> Qkm(x)
    m1 = m * m - 1.0
    m4 = m * m - 4.0
    for ik in range(m + 1, nk):
        c1 = 2.0 * ik - 1.0
        c2 = np.sqrt((ik + 1.0) * (ik - 3.0) - m4)
        c3 = 1.0 / np.sqrt((ik + 1.0) * (ik - 1.0) - m1)
        qk[ik] = (c1 * x * qk[ik - 1] - c2 * qk[ik - 2]) * c3
    return qk


def gauss_zeroes_weights(x1, x2, n):
    const_yeps = 3.0e-14
    x = np.zeros(n)
    w = np.zeros(n)
    m = int((n+1)/2)
    yxm = 0.5*(x2 + x1)
    yxl = 0.5*(x2 - x1)
    for i in range(m):
        yz = np.cos(np.pi*(i + 0.75)/(n + 0.5))
        while True:
            yp1 = 1.0
            yp2 = 0.0
            for j in range(n):
                yp3 = yp2
                yp2 = yp1
                yp1 = ((2.0*j + 1.0)*yz*yp2 - j*yp3 )/(j+1)
            ypp = n*(yz*yp1 - yp2)/(yz*yz - 1.0)
            yz1 = yz
            yz = yz1 - yp1/ypp
            if (np.abs(yz - yz1) < const_yeps):
                break # exit while loop
        x[i] = yxm - yz*yxl
        x[n-1-i] = yxm + yxl*yz
        w[i] = 2.0*yxl/((1.0 - yz*yz)*ypp*ypp)
        w[n-1-i] = w[i]
    return x, w
def Henyey_Greenstein(g,mu,dmu):
    temp = (1 - g ** 2)
    temp1 = (1 + g ** 2 - 2 * g * mu) ** 1.5
    phase_function = temp/temp1
    norm_factor = 0.5 * np.sum(phase_function*dmu)
    phase_function /= norm_factor
    return phase_function


def equal_angle(nAngle,sharp=True):
    angles = np.linspace(0, 180, nAngle)
    step = 180.0 / (nAngle - 1)

    mu_up = np.cos((angles+step/2.0 )* np.pi / 180.0 )
    mu_down = np.cos((angles-step / 2.0) * np.pi / 180.0)

    if not sharp:
        angles[0] += step/4.0
        angles[-1] -= step/4.0
    mu = np.cos(angles * np.pi / 180.0)

    mu_up[-1] = -1
    mu_down[0] = 1
    dmu = np.abs(mu_down-mu_up)
    return np.flip(mu),np.flip(dmu)

def moving_average(data, window_size=3):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')

if __name__ == "__main__":
    g = 0.9
    Lmax = 1000

    x,w = equal_angle(181)

    phase_function = Henyey_Greenstein(g,x,w)
    gl,alpha,_,f11,x2,w2 = Legendre_coefficients(x, phase_function, Lmax)
    nAngle = 181
    mu,dmu = gauss_zeroes_weights(-1, 1, nAngle)

    Pl = legendre_polynomial(mu, Lmax)
    regenerated_phase_function = np.sum(alpha[np.newaxis,:]*Pl,axis=1)
    import matplotlib.pyplot as plt

  
    plt.figure(figsize=(10, 6))
    plt.plot(np.acos(x) / np.pi * 180.0, phase_function, 'r-', linewidth=3)
    plt.plot(np.acos(x2) / np.pi * 180.0, f11, 'k-', linewidth=1)
    plt.plot(np.acos(mu)/np.pi*180.0, regenerated_phase_function, 'b-', linewidth=0.5)
    plt.xlabel('angle', fontsize=12)
    plt.ylabel('phase', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.xticks(np.arange(0, 181, 30))
    plt.tight_layout()
    plt.show()
    print("ok")


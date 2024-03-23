import numpy as np
from scipy.fftpack import dstn, idstn
from scipy.special import erf
import matplotlib.pyplot as plt

T = 85.5
kB = 1.0
beta = 1 / T / kB

amph = 167101.0

pts = 100
r = 14.0
ns = 1

dr = r / pts
dk = 2.0 * np.pi / (2.0 * pts * dr)

rgrid = np.arange(0.5, pts, 1.0) * dr
kgrid = np.arange(0.5, pts, 1.0) * dk

# assert(dr * dk == np.pi/pts)
print(dr * dk, np.pi/pts)


# multiplicity = np.diag([1.0, 1.0, 1.0])
# density = np.diag([0.0334, 0.0334, 0.0334])
# labels = ["O", "H1", "H2"]
# coords = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([-0.333314, 0.942816, 0.0])]
# params = [[78.15, 3.16572, -0.8476], [7.815, 1.16572, 0.4238], [7.815, 1.16572, 0.4238]]

# multiplicity = np.diag([1.0, 2.0])
# density = np.diag([0.0334, 0.0334])
# labels = ["O", "H"]
# coords = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([-0.333314, 0.942816, 0.0])]
# params = [[78.15, 3.16572, -0.8476], [7.815, 1.16572, 0.4238]]

multiplicity = np.diag([1.0])
density = np.diag([0.021017479720736955])
labels = ["Ar"]
coords = [np.array([0.0, 0.0, 0.0])]
params = [[120.0, 3.4, 0.0]]

# multiplicity = np.diag([1.0, 1.0])
# density = np.diag([0.01867, 0.01867])
# labels = ["N1", "N2"]
# coords = [np.array([0.0, 0.0, 0.0]), np.array([1.1, 0.0, 0.0])]
# params = [[44.0, 3.341, 0.0], [44.0, 3.341, 0.0]]

full_ns = int(multiplicity.sum())

dist = np.zeros((full_ns, full_ns))

for i, j in np.ndindex((full_ns, full_ns)):
    dist[i, j] = np.linalg.norm(coords[i] - coords[j])

def wk(dist, k):
    out = np.zeros((pts, ns, ns))
    for i, j in np.ndindex((ns, ns)):
        dists = []
        mult_j = int(multiplicity[j, j])
        dists.append(dist[i, j])
        if mult_j > 1:
            for mj in range(1, mult_j):
                dists.append(dist[i, j+mj])
        for dist_ij in dists:   
            if dist_ij < 0.0:
                out[:, i, j] += np.zeros(pts)
            elif dist_ij == 0.0:
                out[:, i, j] += np.ones(pts)
            else:
                out[:, i, j] += np.sin(k * dist_ij) / (k * dist_ij)
        
        out[:, i, j] /= mult_j

    return out

wk = wk(dist, kgrid)

def lorentz_berthelot(eps1, eps2, sig1, sig2):
    return np.sqrt(eps1 * eps2), 0.5 * (sig1 + sig2)

def lj_impl(epsilon, sigma, r):
    return 4.0 * epsilon * ( np.power(sigma / r, 12.0) - np.power(sigma / r, 6.0) )

def cou_impl(q, r):
    return amph * q / r 

def ur(params, beta, r):
    out = np.zeros((pts, ns, ns))
    for i, j in np.ndindex((ns, ns)):
        eps, sig = lorentz_berthelot(params[i][0], params[j][0], params[i][1], params[j][1])
        q = params[i][2] * params[j][2]
        out[:, i, j] = lj_impl(eps, sig, r) + cou_impl(q, r)

    return out

ur = ur(params, beta, rgrid)

#plt.plot(rgrid, vr[:, 0, 0])
#plt.ylim([-1.0, 1.0])
#plt.show()

def renorm(params, r, k):
    out_r = np.zeros((pts, ns, ns))
    out_k = np.zeros((pts, ns, ns))
    for i, j in np.ndindex((ns, ns)):
        q = params[i][2] * params[j][2]
        out_r[:, i, j] = amph * q * erf(r) / r
        out_k[:, i, j] = 4.0 * np.pi * amph * q * np.exp(-np.power(k, 2.0) / 4.0) / np.power(k, 2.0)

    return out_r,out_k

u_ng_r, u_ng_k = renorm(params, rgrid, kgrid)

ur_sr = ur - u_ng_r

# plt.plot(rgrid, vr_sr[:, 0, 0])
# plt.ylim([-1.0, 1.0])
# plt.show()

def hankel_forward(fr, r, k, dr):
    return 2.0 * np.pi * dr * dstn(fr * r[:, np.newaxis, np.newaxis], type=4, axes=[0]) / k[:, np.newaxis, np.newaxis]

def hankel_inverse(fk, r, k, dk):
    return dk / 4.0 / np.pi / np.pi * idstn(fk * k[:, np.newaxis, np.newaxis], type=4, axes=[0]) / r[:, np.newaxis, np.newaxis]

def HNC(vr_sr, tr):
    return np.exp(-beta * ur_sr + tr) - tr - 1.0

def RISM(cr, wk, n, rho, ur_ng_k):
    hk = np.zeros((pts, ns, ns))
    identity = np.eye(ns)

    ck = hankel_forward(cr, rgrid, kgrid, dr)
    
    ck = ck - beta * ur_ng_k

    for l in np.arange(pts):
        hk[l] =  wk[l] @ (n @ ck[l] @ n) @np.linalg.inv(identity - rho @ wk[l] @ (n @ ck[l]) @ n) @ wk[l]

    tk = (hk - ck) - beta * ur_ng_k

    tr = hankel_inverse(tk, rgrid, kgrid, dk)
    
    return tr

tr = np.zeros((pts, ns, ns))

tol = 1e-5
maxstep = 10000

gamma_min = 0.1
alpha = 2.5
scale = 0.2

iter_count = 0

cr = None

delta_prev = 0.0

gamma = gamma_min

while iter_count < maxstep:
    tr_prev = tr

    cr_loop = HNC(ur_sr, tr)
    tr_curr = RISM(cr_loop, wk, multiplicity, density, u_ng_k)
    
    delta = (tr_curr - tr_prev).max()
    
    if delta > delta_prev:
        gamma_min *= scale
        alpha = -np.log( (gamma * scale - gamma_min) / (1.0 - gamma_min) ) / (delta - tol)
    
    if delta > tol:
        gamma = gamma_min + (1.0 - gamma_min) * np.exp(-alpha * (delta - tol))
    else:
        gamma = 1.0
    
    tr_new = tr_prev + gamma * (tr_curr - tr_prev)
   
    rms = np.sqrt(np.power(tr_new - tr_prev, 2.0).sum() * dr)
    
    tr = tr_new

    print(iter_count, rms, gamma)

    if rms < tol:
        break
    
    iter_count += 1
    
cr = HNC(ur_sr, tr)
    
gr = tr + cr + 1.0

plt.plot(rgrid, gr[:, 0, 0])
# plt.plot(rgrid, gr[:, 0, 1])
# plt.plot(rgrid, gr[:, 0, 2])
# plt.plot(rgrid, gr[:, 1, 1])
# plt.plot(rgrid, gr[:, 1, 2])
# plt.plot(rgrid, gr[:, 2, 2])
plt.show()

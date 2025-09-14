import math
import numpy as np
from numba.experimental import jitclass
from numba import njit, float64, int64

# -----------------------------------------------------------------------------
# SIMD-accelerated helpers and in-place integrators
# -----------------------------------------------------------------------------
@njit(fastmath=True)
def trapz_r2_f_simd(r_grid, fvals):
    r2f = r_grid * r_grid * fvals
    dr = r_grid[1:] - r_grid[:-1]
    mid = 0.5 * (r2f[:-1] + r2f[1:])
    return np.dot(mid, dr)

@njit(fastmath=True)
def trapz_r2_dynamic_simd(r_grid, fvals, n):
    r_n = r_grid[:n]
    f_n = fvals[:n]
    r2f = r_n * r_n * f_n
    dr = r_n[1:] - r_n[:-1]
    mid = 0.5 * (r2f[:-1] + r2f[1:])
    return np.dot(mid, dr)

@njit(fastmath=True)
def select_grid_n(r, r200, max_grid_size):
    min_n = 20
    max_n = max_grid_size
    ratio = r / r200
    if ratio > 4.0:
        return 2 * max_n
    if ratio < 0.0:
        ratio = 0.0
    elif ratio > 1.0:
        ratio = 1.0
    return int(min_n + ratio * (max_n - min_n))

@njit(fastmath=True)
def interp_simd(x, y, xv):
    n = x.size
    i = np.searchsorted(x, xv)
    if i == 0:
        dx = x[1] - x[0]; dy = y[1] - y[0]
        return y[0] + (dy/dx if dx != 0 else 0.0)*(xv - x[0])
    if i >= n:
        dx = x[n-1] - x[n-2]; dy = y[n-1] - y[n-2]
        return y[n-1] + (dy/dx if dx != 0 else 0.0)*(xv - x[n-1])
    x0, x1 = x[i-1], x[i]; y0, y1 = y[i-1], y[i]
    dx = x1 - x0
    return y0 + ((y1 - y0)/(dx if dx != 0 else 1.0))*(xv - x0)

@njit(fastmath=True)
def m_enclosed_nfw_inplace(
    r, r200, rho0_nfw, rs, rt, max_grid_size,
    grid_buf, f_buf
):
    rmin = 1e-4 * r200
    if r <= rmin:
        return 0.0
    n = select_grid_n(r, r200, max_grid_size)
    
    dlog = (math.log(r) - math.log(rmin)) / (n - 1)
    g = math.exp(dlog)

    inv_rs = 1.0 / rs
    inv_rt = 1.0 / rt

    rp = rmin
    grid_buf[0] = rp
    x = rp * inv_rs
    y = rp * inv_rt
    f_buf[0] = (1.0/(x*(1.0+x)*(1.0+x)*(1.0+y*y)*(1.0+y*y)) if x > 0.0 else 0.0)

    for j in range(1, n):
        rp = rp * g
        grid_buf[j] = rp
        x = rp * inv_rs
        y = rp * inv_rt
        f_buf[j] = (1.0/(x*(1.0+x)*(1.0+x)*(1.0+y*y)*(1.0+y*y)) if x > 0.0 else 0.0)
        
    I = trapz_r2_dynamic_simd(grid_buf, f_buf, n)
    return 4.0 * math.pi * rho0_nfw * I

@njit(fastmath=True)
def m_enclosed_gas_inplace(
    r, r200, rco, rej, beta, rho0_gas, max_grid_size,
    grid_buf, f_buf
):
    rmin = 1e-4 * r200
    if r <= rmin:
        return 0.0
    n = select_grid_n(r, r200, max_grid_size)

    dlog = (math.log(r) - math.log(rmin)) / (n - 1)
    g = math.exp(dlog)

    inv_rco = 1.0 / rco
    inv_rej = 1.0 / rej
    b = -beta
    gam = -3.5 + 0.5*beta   # == -(7 - beta)/2

    rp = rmin
    grid_buf[0] = rp
    u = rp * inv_rco
    v = rp * inv_rej
    f_buf[0] = (1.0 + u)**b * (1.0 + v*v)**gam

    for j in range(1, n):
        rp = rp * g
        grid_buf[j] = rp
        u = rp * inv_rco
        v = rp * inv_rej
        f_buf[j] = (1.0 + u)**b * (1.0 + v*v)**gam

    I = trapz_r2_dynamic_simd(grid_buf, f_buf, n)
    return 4.0 * math.pi * rho0_gas * I
    
@njit(fastmath=True)
def m_enclosed_cga(r, fcga, Mtot_nfw, Rh):
    return fcga * Mtot_nfw * math.erf(r/(2.0*Rh)) if r>0 else 0.0

@njit(fastmath=True)
def bisect_zeta(
    r, tol,
    fclm, a, n,
    r200, rho0_nfw, rs, rt, rej,
    rho0_gas, rco, beta,
    fcga, Mtot_nfw, Rh,
    grid_size,
    grid_buf_nfw, f_buf_nfw,
    grid_buf_gas, f_buf_gas
):
    low, high = 0.1, 10.0
    Mg = m_enclosed_gas_inplace(r, r200, rco, rej, beta, rho0_gas, grid_size, grid_buf_gas, f_buf_gas)
    Mc = m_enclosed_cga(r, fcga, Mtot_nfw, Rh)
    niter = int(math.ceil(math.log2((high-low)/tol)))
    for _ in range(niter):
        mid = 0.5*(low+high)
        Mi = m_enclosed_nfw_inplace(r / mid, r200, rho0_nfw, rs, rt, grid_size, grid_buf_nfw, f_buf_nfw)
        Mf = fclm*Mi + Mg + Mc
        fval = mid-1.0 - a*((Mi/Mf)**n - 1.0)
        mask = 1.0*(fval<0.0)
        low  = mask*mid + (1.0-mask)*low
        high = mask*high + (1.0-mask)*mid
    return 0.5*(low+high)

@njit(fastmath=True)
def compute_zeta_grid_inplace(
    r200, tol,
    fclm, a, n,
    rho0_nfw, rs, rt, rej,
    rho0_gas, rco, beta,
    fcga, Mtot_nfw, Rh,
    grid_size, zeta_grid_size,
    zeta_r, zeta_vals,
    grid_buf_nfw, f_buf_nfw,
    grid_buf_gas, f_buf_gas
):
    log_min = math.log(1e-2); log_max = math.log(2*r200)
    inv_d = 1.0/(zeta_grid_size-1)
    for i in range(zeta_grid_size):
        zeta_r[i] = math.exp(log_min + (log_max-log_min)*i*inv_d)
        r = zeta_r[i]
        zeta_vals[i] = bisect_zeta(
            r, tol,
            fclm, a, n,
            r200, rho0_nfw, rs, rt, rej,
            rho0_gas, rco, beta,
            fcga, Mtot_nfw, Rh,
            grid_size,
            grid_buf_nfw, f_buf_nfw,
            grid_buf_gas, f_buf_gas
        )

@njit(fastmath=True)
def compute_normalizations_simd(
    M200, r200, rs, rt, rej,
    fgas, fcga, Rh, grid_size,
    rco, beta,
    grid_buf_nfw, f_buf_nfw,
    grid_buf_gas, f_buf_gas
):
    pi = math.pi
    integral_nfw = m_enclosed_nfw_inplace(r200, r200, 1.0, rs, rt, grid_size, grid_buf_nfw, f_buf_nfw)
    rho0_nfw = M200/(integral_nfw)
    r_upper = 10.0*max(rt, rej)
    Mtot_nfw = m_enclosed_nfw_inplace(r_upper, r200, rho0_nfw, rs, rt, grid_size, grid_buf_nfw, f_buf_nfw)
    integral_gas = m_enclosed_gas_inplace(r_upper, r200, rco, rej, beta, 1.0, grid_size, grid_buf_gas, f_buf_gas)
    rho0_gas = fgas*Mtot_nfw/(integral_gas)
    norm_cga = fcga*Mtot_nfw/(4.0*pi**1.5*Rh)
    return rho0_nfw, Mtot_nfw, rho0_gas, norm_cga

@njit(fastmath=True)
def m_enclosed_clm_simd(
    r, r200, rho0_nfw, rs, rt, grid_size,
    fclm, zeta_r, zeta_vals,
    grid_buf_nfw, f_buf_nfw,
    grid_buf_gas, f_buf_gas
):
    if r <= 0.0:
        return 0.0
    z = interp_simd(zeta_r, zeta_vals, r)
    m_nfw = m_enclosed_nfw_inplace(r/ z, r200, rho0_nfw, rs, rt, grid_size, grid_buf_nfw, f_buf_nfw)
    return fclm * m_nfw

@njit(fastmath=True)
def m_enclosed_simd(
    r,
    r200, rho0_nfw, rs, rt, grid_size,
    rco, rej, beta, rho0_gas,
    fclm, fcga, Mtot_nfw, Rh,
    zeta_r, zeta_vals,
    grid_buf_nfw, f_buf_nfw,
    grid_buf_gas, f_buf_gas
):
    if r <= 0.0:
        return 0.0
    mgas = m_enclosed_gas_inplace(r, r200, rco, rej, beta, rho0_gas, grid_size, grid_buf_gas, f_buf_gas)
    mcga = m_enclosed_cga(r, fcga, Mtot_nfw, Rh)
    mclm = m_enclosed_clm_simd(r, r200, rho0_nfw, rs, rt, grid_size, fclm, zeta_r, zeta_vals, grid_buf_nfw, f_buf_nfw, grid_buf_gas, f_buf_gas)
    return mgas + mcga + mclm

@njit(fastmath=True)
def radius_from_random_simd(
    u, M200, r200,
    rho0_nfw, rs, rt, grid_size,
    rco, rej, beta, rho0_gas,
    fclm, fcga, Mtot_nfw, Rh,
    zeta_r, zeta_vals,
    grid_buf_nfw, f_buf_nfw,
    grid_buf_gas, f_buf_gas,
    tol
):
    target = u*M200
    low, high = 0.0, 2.0*r200
    niter = int(math.ceil(math.log2((high-low)/tol)))
    for _ in range(niter):
        mid = 0.5*(low+high)
        mval = m_enclosed_simd(
            mid, r200, rho0_nfw, rs, rt, grid_size,
            rco, rej, beta, rho0_gas,
            fclm, fcga, Mtot_nfw, Rh,
            zeta_r, zeta_vals,
            grid_buf_nfw, f_buf_nfw,
            grid_buf_gas, f_buf_gas
        )
        mask = 1.0*(mval<target)
        low  = mask*mid + (1.0-mask)*low
        high = mask*high + (1.0-mask)*mid
    return 0.5*(low+high)

# -----------------------------------------------------------------------------
# BCMProfile class
# -----------------------------------------------------------------------------
spec = [
    ('M200', float64), ('r200', float64), ('c', float64),
    ('theta_ej', float64), ('theta_co', float64),
    ('Mc', float64), ('mu', float64),
    ('A_star', float64), ('M1_star', float64),
    ('eta_star', float64), ('eta_cga', float64),
    ('eps', float64), ('Omega0_b', float64), ('Omega0_M', float64),
    ('a_dm', float64), ('n_dm', float64),
    ('grid_size', int64), ('zeta_grid_size', int64), ('tol', float64),
    # derived
    ('rs', float64), ('rt', float64), ('rej', float64), ('rco', float64), ('beta', float64), ('Rh', float64),
    ('fgas', float64), ('fclm', float64), ('fcga', float64), ('fstar', float64),
    ('rho0_nfw', float64), ('Mtot_nfw', float64), ('rho0_gas', float64), ('norm_cga', float64),
    # buffers
    ('zeta_r', float64[:]), ('zeta_vals', float64[:]),
    ('grid_buf_nfw', float64[:]), ('f_buf_nfw', float64[:]),
    ('grid_buf_gas', float64[:]), ('f_buf_gas', float64[:])
]

@jitclass(spec)
class BCMProfile:
    def __init__(self,
                 M200, r200, c,
                 theta_ej, theta_co,
                 Mc, mu,
                 A_star, M1_star, eta_star, eta_cga,
                 eps, Omega0_b, Omega0_M,
                 a_dm, n_dm,
                 grid_buf_nfw, f_buf_nfw,
                 grid_buf_gas, f_buf_gas,
                 zeta_r, zeta_vals,
                 grid_size, zeta_grid_size, tol):
        self.M200 = M200; self.r200 = r200; self.c = c
        self.theta_ej = theta_ej; self.theta_co = theta_co
        self.Mc = Mc; self.mu = mu
        self.A_star = A_star; self.M1_star = M1_star
        self.eta_star = eta_star; self.eta_cga = eta_cga
        self.eps = eps; self.Omega0_b = Omega0_b; self.Omega0_M = Omega0_M
        self.a_dm = a_dm; self.n_dm = n_dm
        self.grid_size = grid_size; self.zeta_grid_size = zeta_grid_size; 
        self.tol = tol

        self.zeta_r = zeta_r
        self.zeta_vals = zeta_vals
        self.grid_buf_nfw = grid_buf_nfw
        self.f_buf_nfw    = f_buf_nfw
        self.grid_buf_gas = grid_buf_gas
        self.f_buf_gas    = f_buf_gas
        
        self._compute_derived()
        self._compute_normalizations()
        self._build_zeta_lookup()

    def _compute_derived(self):
        self.rs = self.r200/self.c
        self.rt = self.eps*self.r200
        self.rej = self.theta_ej*self.r200
        self.rco = self.theta_co*self.r200
        self.beta = 3.0 - (self.Mc/self.M200)**self.mu
        f_dm0 = (self.Omega0_M - self.Omega0_b)/self.Omega0_M
        self.fstar = self.A_star*(self.M1_star/self.M200)**self.eta_star
        self.fgas = self.Omega0_b/self.Omega0_M - self.fstar
        self.fcga = self.A_star*(self.M1_star/self.M200)**self.eta_cga
        self.fclm = f_dm0 + (self.fstar - self.fcga)
        self.Rh = 0.015*self.r200

    def _compute_normalizations(self):
        self.rho0_nfw, self.Mtot_nfw, self.rho0_gas, self.norm_cga = \
            compute_normalizations_simd(
                self.M200, self.r200, self.rs, self.rt, self.rej,
                self.fgas, self.fcga, self.Rh, self.grid_size,
                self.rco, self.beta,
                self.grid_buf_nfw, self.f_buf_nfw,
                self.grid_buf_gas, self.f_buf_gas
            )

    def _build_zeta_lookup(self):
        compute_zeta_grid_inplace(
            self.r200, self.tol,
            self.fclm, self.a_dm, self.n_dm,
            self.rho0_nfw, self.rs, self.rt, self.rej,
            self.rho0_gas, self.rco, self.beta,
            self.fcga, self.Mtot_nfw, self.Rh,
            self.grid_size, self.zeta_grid_size,
            self.zeta_r, self.zeta_vals,
            self.grid_buf_nfw, self.f_buf_nfw,
            self.grid_buf_gas, self.f_buf_gas
        )

    def reset_halo(self, M200, r200, c):

        self.M200 = M200; self.r200 = r200; self.c = c
        self._compute_derived()
        self._compute_normalizations()
        self._build_zeta_lookup()    
    
    def sample_radius(self, u):
        return radius_from_random_simd(
            u, self.M200, self.r200,
            self.rho0_nfw, self.rs, self.rt, self.grid_size,
            self.rco, self.rej, self.beta, self.rho0_gas,
            self.fclm, self.fcga, self.Mtot_nfw, self.Rh,
            self.zeta_r, self.zeta_vals,
            self.grid_buf_nfw, self.f_buf_nfw,
            self.grid_buf_gas, self.f_buf_gas,
            self.tol
        )

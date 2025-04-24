from classy import Class
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline
import numpy as np
import os

os.makedirs("plots", exist_ok=True)

cosmo = Class()
cosmo.set({
    'omega_b':      0.02238,      # physical baryon density
    'omega_cdm':    0.11600,      # physical cold dark matter density
    'h':            0.68,         # reduced hubble parameter
    'A_s':          2.1e-9,       # primordial scalar amplitude
    'n_s':          0.97,         # scalar spectral index
    'tau_reio':     0.054,        # reionization optical depth
})
cosmo.set({'output':'tCl,pCl,lCl,nCl,mPk','lensing':'yes','P_k_max_1/Mpc': 3.0, 'z_max_pk': 100})
cosmo.compute()

kk = np.logspace(-4,np.log10(3),1000) # k in h/Mpc
Pk = [] # P(k) in (Mpc/h)**3
h = 0.68 # for conversions to 1/Mpc
for k in kk:
    Pk.append(cosmo.pk(k*h,0.)*h**3) # function .pk(k,z)

Pk = np.array(Pk)

delta = 0.03  # spline spacing
k_min, k_max = kk.min(), kk.max()
knots = np.arange(k_min + delta, k_max - delta, delta)

spline = LSQUnivariateSpline(kk, Pk, knots, k=3)
P_bb = spline(kk) # no wiggle
P_wig = Pk - P_bb # wiggle-only

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1]},
    figsize=(8, 6)
)

# top panel
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(kk[0], kk[-1])

ax1.plot(kk, Pk,    label=r'$P(k)$')
ax1.plot(kk, P_bb,  '--', label=r'$P_{\rm smooth}(k)$')

ax1.set_ylabel(r'$P(k)\ [(\mathrm{Mpc}/h)^3]$')
ax1.legend(loc='best')
ax1.set_title('Simulated power spectrum')

# bottom pannel
ax2.plot(kk, P_wig, '-', label=r'$P_{\rm wig}(k)$')
ax2.set_xscale('log')  # keep the same log‐x axis
ax2.set_xlabel(r'$k\ [h/\mathrm{Mpc}]$')
ax2.set_ylabel(r'$\Delta P$')
ax2.legend(loc='best')
ax2.set_title('Isolated oscillations')


plt.tight_layout()
fig.savefig('plots/pk.pdf')
# plt.show()

r = np.linspace(1, 160, 400)  # r in Mpc/h; domain chosen mostly arbitrarily

xi_wig = np.zeros_like(r)
xi = np.zeros_like(r)

for i, ri in enumerate(r):
    kr = kk * ri
    # sin(kr)/(kr) = sinc(kr/pi)
    
    window = 0.5 * ( 1 - np.tanh( 2 * (kk - kk[-1] + 1.5) ) ) # Lepori et. al: "...we introduced a cutoff W to smooth numerical spurious oscillations..."
    integrand = kk**2 * Pk * np.sinc(kr / np.pi) * window
    xi[i] = np.trapz(integrand, kk)
    
    integrand_wig = kk**2 * P_wig * np.sinc(kr / np.pi)
    xi_wig[i] = np.trapz(integrand_wig, kk)

xi_wig /= 2.0 * np.pi**2
xi /= 2.0 * np.pi**2

# xi plot
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    sharex=True,
    gridspec_kw={'height_ratios': [3, 2]},
    figsize=(8, 6)
)

# top panel
ax1.plot(r, xi * r**2, '-', label=r'$r^2\,\xi(r)$')
ax1.set_ylabel(r'$r^2\,\xi(r)\,[(\mathrm{Mpc}/h)^2]$')
ax1.set_title('2-point correlation function')
ax1.legend(loc='best')
ax1.grid(ls='--', alpha=0.5)

# bottom panel
ax2.plot(r, xi_wig * r**2, '-', label=r'$r^2\,\xi_{\rm wig}(r)$')
ax2.set_xlabel(r'$r\ [\mathrm{Mpc}/h]$')
ax2.set_ylabel(r'$r^2\,\xi_{\rm wig}(r)$')
ax2.legend(loc='best')
ax2.grid(ls='--', alpha=0.5)


plt.tight_layout()
fig.savefig('plots/xi.pdf')
# plt.show()

r_peak = r[np.argmax(xi_wig)]

c = 3e5  # km/s

def E(z,Om,Ok,w):
    return np.sqrt(Om*(1+z)**3 + Ok*(1+z)**2 + (1-Om-Ok)*(1+z)**(3*(1+w)))

def H(z,Om,Ok,w,H0=70):
    return H0 * E(z,Om,Ok,w)

def comoving_distance(z,Om,Ok,w,H0=70):
    def dc(z_i):
        zs = np.linspace(0,z_i,500)
        return (c/H0)*np.trapz(1/E(zs,Om,Ok,w),zs)
    return np.array([dc(z_i) for z_i in np.atleast_1d(z)])

def angular_diameter_distance(z,Om,Ok,w,H0=70):
    Dc = comoving_distance(z,Om,Ok,w,H0)
    if Ok>0:
        a = np.sqrt(Ok)
        return (c/H0)/a * np.sinh(a*Dc*H0/c)/(1+z)
    elif Ok<0:
        a = np.sqrt(-Ok)
        return (c/H0)/a * np.sin(a*Dc*H0/c)/(1+z)
    else:
        return Dc/(1+z)

# precomputed BAO-band centers & measurement:
z_vals   = np.linspace(0.7,1.8,7)
delta_z  = r_peak * H(z_vals,0.3,0.0,-1.0)/c
theta    = r_peak / ((1+z_vals)*angular_diameter_distance(z_vals,0.3,0.0,-1.0))
F_meas   = delta_z/theta
sigma    = 0.02*F_meas

def log_prior(p):
    Om,Ok,w = p
    return 0.0 if (0<Om<1 and -0.5<Ok<0.5 and -2<w<0) else -np.inf

def log_likelihood(p):
    Om,Ok,w = p
    F_th = (1+z_vals)*angular_diameter_distance(z_vals,Om,Ok,w)*H(z_vals,Om,Ok,w)/c
    return -0.5*np.sum(((F_meas-F_th)/sigma)**2)

def log_post(p):
    lp = log_prior(p)
    return lp + log_likelihood(p) if np.isfinite(lp) else -np.inf


n_chains = 4
nsteps   = 100_000
burn     =   10_000
thin     =      100

prop_scale = np.array([0.02,0.02,0.1])
all_samples = []

for chain_id in range(n_chains):
    np.random.seed(chain_id)  # different seed per chain
    chain = np.zeros((nsteps,3))
    chain[0] = [0.3 + 0.05*chain_id,  # stagger inits
                0.0, 
               -1.0]
    curr_lp = log_post(chain[0])
    for i in range(1,nsteps):
        if i % 20000 == 0: print(i)
        prop = chain[i-1] + prop_scale*np.random.randn(3)
        lp   = log_post(prop)
        if np.log(np.random.rand()) < (lp-curr_lp):
            chain[i], curr_lp = prop, lp
        else:
            chain[i] = chain[i-1]
    # thin + burn
    thinned = chain[burn::thin]
    all_samples.append(thinned)

# stack: shape (n_chains, n_samples_thin, 3)
samples_arr = np.stack(all_samples, axis=0)

# compute Gelman–Rubin R̂
m, n, d = samples_arr.shape
Rhat = np.zeros(d)
for j in range(d):
    chain_means = samples_arr[:,:,j].mean(axis=1)
    overall_mean = chain_means.mean()
    # between-chain variance
    B = n/(m-1) * np.sum((chain_means - overall_mean)**2)
    # within-chain variance
    W = 1/(m*(n-1)) * np.sum((samples_arr[:,:,j] - chain_means[:,None])**2)
    var_hat = (1 - 1/n)*W + (1/n)*B
    Rhat[j] = np.sqrt(var_hat / W)

print()
print("Gelman–Rubin R:")
for lbl, r in zip([r"Ωₘ", r"Ωₖ", r"w"], Rhat):
    print(f"  {lbl} →  R̂ = {r:.3f}") # want <= 1.1

flat_samples = samples_arr.reshape(-1,3)


# corner‐style scatter‐hist
labels = [r"$\Omega_m$", r"$\Omega_k$", r"$w$"]
fig, axes = plt.subplots(3,3,figsize=(9,9))
for i in range(3):
    for j in range(3):
        ax = axes[i,j]
        if i==j:
            ax.hist(flat_samples[:,i], bins=40, color='0.7')
            ax.set_xlabel(labels[i])
        elif i>j:
            ax.scatter(flat_samples[:,j], flat_samples[:,i], s=2, alpha=0.2)
            ax.set_xlabel(labels[j])
            ax.set_ylabel(labels[i])
        else:
            ax.axis('off')
plt.tight_layout()
plt.savefig("plots/parameter_corner.pdf")
# plt.show()

means = flat_samples.mean(axis=0)
stds  = flat_samples.std(axis=0, ddof=1)

for name, μ, σ in zip(["Ωₘ","Ωₖ","w"], means, stds):
    print(f"{name} = {μ:.3f} ± {σ:.3f}")


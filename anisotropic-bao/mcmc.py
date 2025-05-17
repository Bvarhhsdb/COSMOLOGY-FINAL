import numpy as np
import matplotlib.pyplot as plt

c = 3e5
def E(z,Om,Ok,w):
    return np.sqrt(Om*(1+z)**3 + Ok*(1+z)**2 + (1-Om-Ok)*(1+z)**(3*(1+w)))
def H(z,Om,Ok,w,H0=70):
    return H0 * E(z,Om,Ok,w)
def comoving_distance(z,Om,Ok,w,H0=70):
    def dc(z_i):
        zs = np.linspace(0,z_i,500)
        return (c/H0)*np.trapz(1/E(zs,Om,Ok,w),zs)
    return np.array([dc(zi) for zi in np.atleast_1d(z)])
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

r = np.linspace(50,150,500)
theta = np.linspace(1,3,500)*np.pi/180
xi_rad = np.loadtxt('xi_rad.txt')
xi_trans = np.loadtxt('xi_trans.txt')
r_peak = r[np.argmax(xi_rad)]
theta_peak = theta[np.argmax(xi_trans)]

z_vals = np.linspace(0.7,1.8,7)
delta_z = r_peak * H(z_vals,0.3,0.0,-1.0)/c
theta_obs = theta_peak/((1+z_vals)*angular_diameter_distance(z_vals,0.3,0.0,-1.0))
F_meas = delta_z/theta_obs
sigma = 0.02*F_meas

def log_prior(p):
    Om,Ok,w = p
    return 0.0 if (0<Om<1 and -0.5<Ok<0.5 and -2<w<0) else -np.inf
def log_like(p):
    Om,Ok,w = p
    F_th = (1+z_vals)*angular_diameter_distance(z_vals,Om,Ok,w)*H(z_vals,Om,Ok,w)/c
    return -0.5*np.sum(((F_meas-F_th)/sigma)**2)
def log_post(p):
    lp = log_prior(p)
    return lp + log_like(p) if np.isfinite(lp) else -np.inf

n_chains,nsteps,burn,thin = 4,100000,10000,100
prop_scale = np.array([0.02,0.02,0.1])
all_samples = []

for cid in range(n_chains):
    np.random.seed(cid)
    chain = np.zeros((nsteps,3))
    chain[0] = [0.3+0.05*cid,0.0,-1.0]
    curr_lp = log_post(chain[0])
    for i in range(1,nsteps):
        prop = chain[i-1] + prop_scale*np.random.randn(3)
        lp = log_post(prop)
        if np.log(np.random.rand()) < lp-curr_lp:
            chain[i],curr_lp = prop,lp
        else:
            chain[i] = chain[i-1]
    all_samples.append(chain[burn::thin])

samples_arr = np.vstack(all_samples)

labels = [r"$\Omega_m$",r"$\Omega_k$",r"$w$"]
fig,axes = plt.subplots(3,3,figsize=(9,9))
for i in range(3):
    for j in range(3):
        ax=axes[i,j]
        if i==j:
            ax.hist(samples_arr[:,i],bins=40,color='0.7')
            ax.set_xlabel(labels[i])
        elif i>j:
            ax.scatter(samples_arr[:,j],samples_arr[:,i],s=2,alpha=0.2)
            ax.set_xlabel(labels[j]);ax.set_ylabel(labels[i])
        else:
            ax.axis('off')
plt.tight_layout()
plt.savefig("plots/parameter_corner.pdf")

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import math

# Constants
parsec = 3.08567758128 * 1e16
year = 31536000 * 299792458 / parsec
mass_to_length = 1.3466 * 1e27

Msun = 1.98992 * 1e30  # Mass of the Sun in kilograms

# Orbital parameters
Msun_parsec = Msun / (mass_to_length * parsec)
mass_big_parsec = Msun_parsec * 1e3
mass_small_parsec = Msun_parsec
mu_parsec = (mass_big_parsec * mass_small_parsec) / (mass_big_parsec + mass_small_parsec)
m_parsec = mass_big_parsec + mass_small_parsec

# Parameter values
alpha_spike_values = [7/3, 7/4, 9/4]  # Change the value of alpha_spike here
po_6_values = [0, 5.448e15*Msun_parsec / parsec**3, 5.448e17*Msun_parsec / parsec**3]  # Change the value of dark matter density here

r_isco = 6 * mass_big_parsec

# Dark matter density function
def po_dm(r, alpha_spike, po_6):
    r_6 = 1e-6
    return po_6 * (r_6 / r)**alpha_spike

# Differential equations (gravitational waves and dynamical friction)
def dEdt_gw(e, a):
    return -32. * ((mu_parsec**2 * m_parsec**3) * (1. + 73./24.*(e**2) + 37./96.*(e**4))) / (5. * (a**5) * ((1. - e**2)**(7./2.)))

def dLdt_gw(e, a):
    return -32. * ((mu_parsec**2 * m_parsec**(5./2.)) * (1. + 7./8. * e**2)) / (5. * a**(7./2.) * (1 - e**2)**2)

def F(r, v, po_6, alpha_spike):
    ln_lambda = 0.5*np.log(mass_big_parsec / mass_small_parsec)
    v_squared = max(v[0]**2 + v[1]**2, 1e-10)  # Avoid division by zero
    return (4.*np.pi * mass_small_parsec**2 * po_dm(r, alpha_spike, po_6) * ln_lambda) / v_squared

# Main system of equations
def dEdt(e, a, po_6, alpha_spike):
    return dEdt_gw(e, a) + dEdt_df(e, a, po_6, alpha_spike)

def dLdt(e, a, po_6, alpha_spike):
    return dLdt_gw(e, a) + dLdt_df(e, a, po_6, alpha_spike)

def Lorb(e, m_parsec, mu_parsec, Eorb):
    return np.sqrt(((e**2 - 1.) * (m_parsec**2 * mu_parsec**3)) / (2. * Eorb))

def dedt(e, a, po_6, alpha_spike):
    Eorb = -(m_parsec * mu_parsec) / (2. * a)
    Lorb_val = Lorb(e, m_parsec, mu_parsec, Eorb)
    return -((1. - e**2) / (2. * e)) * (dEdt(e, a, po_6, alpha_spike) / Eorb + 2. * dLdt(e, a, po_6, alpha_spike) / Lorb_val)

def dadt(e, a, po_6, alpha_spike):
    dEda = (mass_big_parsec * mass_small_parsec) / (2. * a**2)
    return ((dEdt(e, a, po_6, alpha_spike)) / dEda)

def system_of_equations(t, y, po_6, alpha_spike):
    e, a = y
    dedt_val = dedt(e, a, po_6, alpha_spike)
    dadt_val = dadt(e, a, po_6, alpha_spike)
    return [dedt_val, dadt_val]

def solve_system(e_initial, a_initial, po_6, alpha_spike):
    Tmax = (5. / 256. * a_initial**4) / (mass_big_parsec * mass_small_parsec * (mass_small_parsec + mass_big_parsec))
    t_span = (0, Tmax)
    sol = integrate.solve_ivp(system_of_equations, t_span, [e_initial, a_initial], args=(po_6, alpha_spike), method='RK45', rtol=1e-8, atol=1e-8)
    e_vals, a_vals = sol.y
    return e_vals, a_vals, sol.t

# Insert from code
def get_orbital_elements(e, a, phi):
    r = a*(1. - e**2)/(1. + e*np.cos(phi))
    v = np.sqrt(m_parsec *(2./r - 1./a))
    v_phi = r * np.sqrt(m_parsec * a * (1.-e**2)) / r**2
    v_r = np.sqrt(np.max([v**2 - v_phi**2, 0.]))
    return r, v, v_r, v_phi

def omega_s(r):
    return np.sqrt((mass_big_parsec + mass_small_parsec)/r**3)

def dEdt_df(e, a, po_6, alpha_spike):
    def integrand(phi):
        r, v, v_r, v_phi = get_orbital_elements(e, a, phi)
        return F(r, (v_r, v_phi), po_6, alpha_spike) * v / (1. + e * np.cos(phi))**2

    result, error = integrate.quad(integrand, 0., 2. * np.pi, limit=1000, epsabs=1e-8, epsrel=1e-8)
    return -(1. - e**2)**(3. / 2.) / (2. * np.pi) * result

def dLdt_df(e, a, po_6, alpha_spike):
    def integrand(phi):
        r, v, v_r, v_phi = get_orbital_elements(e, a, phi)
        return F(r, (v_r, v_phi), po_6, alpha_spike) / v / (1. + e * np.cos(phi))**2

    result, error = integrate.quad(integrand, 0., 2. * np.pi, limit=1000, epsabs=1e-8, epsrel=1e-8)
    return -(1. - e**2)**(3. / 2.) / (2. * np.pi) * np.sqrt(m_parsec * a * (1. - e**2)) * result

# Plots for three values of alpha_spike and po_6
plt.figure(figsize=(10, 6))

for alpha_spike in alpha_spike_values:
   po_6 = 5.448e15*Msun_parsec  # Change the value of po_6 here
   e_vals, a_vals, t_vals = solve_system(0.1, 100 * r_isco, po_6, alpha_spike)

   plt.plot(t_vals, np.log10(a_vals / mass_big_parsec), label=f'alpha = {alpha_spike:.2f}')
    
plt.xlabel('Time')
plt.ylabel('log a/M')
plt.xscale('log')  # Set logarithmic scale for time
plt.title(r'$e_0 = 0.10, a_0 = 100.0$')
plt.legend()
plt.grid(True)

plt.show()

po_6 = 5.448e15*Msun_parsec  # Adjust as needed

plt.figure(figsize=(10, 6))

for alpha_spike in alpha_spike_values:  # Loop over different alpha_spike values
    e_vals, a_vals, t_vals = solve_system(0.5, 100 * r_isco, po_6, alpha_spike)
    plt.plot(a_vals / r_isco, e_vals, label=f'α = {alpha_spike:.2f}')  

plt.ylabel('Eccentricity')
plt.xlabel(r'$a / r_{\rm isco}$')
plt.title(r'$e_0 = 0.50, a_0 = 100.0$')
plt.legend()
plt.grid(True)

plt.show()

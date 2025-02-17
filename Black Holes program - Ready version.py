import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import math

parsec = 3.08567758128 * 1e16
year = 31536000*299792458/parsec
mass_to_length = 1.3466*1e27

G = 1  
c = 1  
Msun = 1.98992*1e30  #Sun mass in kilograms

Msun_parsec = Msun/(mass_to_length * parsec)
mass_big_parsec = Msun_parsec*1e3
mass_small_parsec = Msun_parsec
mu_parsec = (mass_big_parsec * mass_small_parsec) / (mass_big_parsec + mass_small_parsec)
m_parsec = mass_big_parsec + mass_small_parsec

r_isco = 6*mass_big_parsec

#Differential equations
def dEdt(e, a):
    return -32.*((mu_parsec**2 * m_parsec**3)*(1.+73./24.*(e**2)+37./96.*(e**4)))/(5.*(a**5)*((1.-e**2)**(7./2.)))

def dLdt(e, a):
    return -32.*((mu_parsec**2 * m_parsec**(5./2.))*(1.+7./8.*e**2))/(5.*a**(7./2.)*(1-e**2)**2)

def Lorb(e, m_parsec, mu_parsec, Eorb):
    return np.sqrt(((e**2-1.)*(m_parsec**2 * mu_parsec**3))/(2.*Eorb))

def dedt(e, a):
    Eorb = -(m_parsec*mu_parsec)/(2.*a) 
    Lorb_val = Lorb(e, m_parsec, mu_parsec, Eorb)
    return -((1.-e**2)/(2*e))*(dEdt(e, a)/Eorb+2*dLdt(e, a)/Lorb_val)

def dadt(e, a):
    dEda = (mass_big_parsec*mass_small_parsec)/(2.*a**2) 
    return ((dEdt(e, a))/(dEda))

def system_of_equations(t, y):
    e, a = y
    dedt_val = dedt(e, a)
    dadt_val = dadt(e, a)
    return [dedt_val, dadt_val]

def solve_system(e_initial, a_initial):
    t_span = (0, 1000*year)  #time interval
    sol = integrate.solve_ivp(system_of_equations, t_span, [e_initial, a_initial], method='RK45', rtol = 1e-10, atol = 1e-10)
    e_vals, a_vals = sol.y
    return e_vals, a_vals, sol.t


e, a, t = solve_system(0.5, 100*r_isco)



plt.figure(figsize=(10, 6))
plt.plot(a/r_isco, e, label='e(a)')
plt.ylabel('Eccentricity')
plt.xlabel('a/r_isco')
plt.title('e_0 = 0.50, a_0 = 100.0')
plt.legend()
plt.grid(True)
plt.show()


t_values = np.linspace(0, 1000*year, len(a))

plt.figure(figsize=(10, 6))
plt.plot(t/year, a/r_isco, label='a(t)')
plt.xscale("log")
plt.yscale("log")
plt.ylim(1,100)
plt.ylabel('log a/M')
plt.xlabel('time')
plt.title('e_0 = 0.10, a_0 = 100.0')
plt.legend()
plt.grid(True)
plt.show()
import numpy as np
import scipy.integrate as nint
import matplotlib.pyplot as plt

m = 1e20
G = 6.67e-8
M = 8.6e30
R =2.5e9
Q = 7.2e4

def a_dot(t,a):
    return -(9/2)*a**(-11/2)*np.sqrt(G/M)*m*R**5/Q

def a(t, a_0):
    factor = -(117/4)*np.sqrt(G/M)*m*R**5/Q
    return ((factor*t + a_0**(13/2))**2)**(1/13)

t = 1.6e17
time = np.linspace(-t,0,1000)

a0 = 2*R
print(a0)

semiMajors = []
for tau in time:
    semiMajors.append(a(tau,a0))

intResult = nint.solve_ivp(a_dot,(-t,0),[a0])


fig,ax = plt.subplots(1,1,figsize = (6,4))
ax.plot(intResult['t'], intResult['y'][0])
ax.plot(time, semiMajors)
ax.hlines(5e9, -np.log(t),1, color = "tab:pink")
plt.savefig("./migration.png")

print(semiMajors[0])

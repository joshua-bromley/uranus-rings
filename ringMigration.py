import numpy as np
from scipy import integrate as nint
from matplotlib import pyplot as plt

G = 4*np.pi**2
M = 4.34e-5
J2 = 3343.43e-6
R = 1.6908e-4
a = 10*R
Ms = 1
ap = 19.165
n_p = [0,0,1]
n_s = [np.sin(np.deg2rad(97.77)),0,np.cos(np.deg2rad(97.77))]




def ddt(t, je):
    j = np.array(je[:3])
    e = np.array(je[3:])

    coeff1 = ((3*np.sqrt(G*M)*J2*R*R)/(2*a**(7/2)))
    coeff2 = 0#((3*np.sqrt(G)*Ms*a**(3/2))/(4*np.sqrt(M)*ap**3))

    

    dj = coeff1*(np.dot(j,n_p))*(np.cross(j,n_p)) + coeff2*(np.dot(j,n_s)*(np.cross(j,n_s))-5*(np.dot(e,n_s)*(np.cross(e,n_s))))
    de = coeff1*((1-5*np.dot(j,n_p)**2)*(np.cross(e,j)) + 2*np.dot(j,n_p)*(np.cross(e,n_p))) + coeff2*(np.dot(j,n_s)*(np.cross(e,n_s)) - 5*np.dot(e,n_s)*(np.cross(j,n_s))+2*(np.cross(j,e)))

    return [dj[0],dj[1],dj[2],de[0],de[1],de[2]]

j0 = [0.1,0,np.sqrt(0.99)]
e0 = [0,0,0]
je0 = np.concatenate((j0,e0))

sol = nint.solve_ivp(ddt,(0, 1e7),je0)

j = np.transpose(sol['y'][:3])
e = np.transpose(sol['y'][3:])
t = sol["t"]

n = np.ones_like(j)
u = np.ones_like(e)
Omega = np.ones(len(n))
omega = np.ones(len(n))
pomega = np.ones(len(n))



for i in range(len(j)):
    u[i] = e[i]/(np.sqrt(np.dot(e[i],e[i]))) if np.dot(e[i],e[i]) != 0 else [0,0,0] 
    n[i] = j[i]/(np.sqrt(1-np.dot(e[i],e[i])**2))
    Omega[i] = np.arccos(np.dot(n[i],[1,0,0]))
    omega[i] = np.arccos(np.dot(n[i],u[i]))
    pomega[i] = Omega[i] + omega[i]

magJ = np.sqrt(j[0]**2 + j[1]**2 + j[2]**2)



fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(n[-1000:,0], n[-1000:,1], n[-1000:,2])
ax.plot([0,n_p[0]],[0,n_p[1]],[0,n_p[2]])
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_zlim(-2,2)
ax.view_init(15,90)
fig.savefig("./ringMigration.png")



fig2, ax2 = plt.subplots(1,1)
ax2.plot(t,Omega, marker = "o", label ="$\Omega$")
ax2.plot(t,omega, marker = "o", label = "$\omega$")
ax2.plot(t,pomega, marker = "o", label = "$\\varpi$")
ax2.legend()

fig2.savefig("./ringMigration2D.png")







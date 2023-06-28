import numpy as np
from scipy import integrate as nint
from tqdm import tqdm
from matplotlib import pyplot as plt

G = 6.67e-11
M = 86.811e24
a = 3e5
J2 = 3343.43e-6
R = 25362e3
Ms = 2e30
ap = 2867.043e9
n_p = [np.sin(np.deg2rad(97.77)),0,np.cos(np.deg2rad(97.77))]
ns = [0,0,1]


coeff1 = ((3*np.sqrt(G*M)*J2*R*R)/(2*a**(7/2)))
coeff2 = ((3*np.sqrt(G)*Ms*a**(3/2))/(4*np.sqrt(M)*ap**3))

def ddt(t, je):
    j = je[:3]
    e = je[3:]
    

    

    dj1 = coeff1*(np.dot(j,n_p))*(j[1]*n_p[2]-j[2]*n_p[1]) + coeff2*(np.dot(j,ns)*(j[1]*ns[2]-j[2]*ns[1])-5*(np.dot(e,ns)*(e[1]*ns[2]-e[2]*ns[1])))
    dj2 = coeff1*(np.dot(j,n_p))*(j[2]*n_p[0]-j[0]*n_p[2]) + coeff2*(np.dot(j,ns)*(j[2]*ns[0]-j[0]*ns[2])-5*(np.dot(e,ns)*(e[2]*ns[0]-e[0]*ns[2])))
    dj3 = coeff1*(np.dot(j,n_p))*(j[0]*n_p[1]-j[1]*n_p[0]) + coeff2*(np.dot(j,ns)*(j[0]*ns[1]-j[1]*ns[0])-5*(np.dot(e,ns)*(e[0]*ns[1]-e[1]*ns[0])))

    de1 = coeff1*((1-5*np.dot(j,n_p)**2)*(e[1]*j[2]-e[2]*j[1]) + 2*np.dot(j,n_p)*(e[1]*n_p[2]-e[2]*n_p[1])) + coeff2*(np.dot(j,ns)*(e[1]*ns[2]-e[2]*ns[1]) - 5*np.dot(e,ns)*(j[1]*ns[2]-j[2]*ns[1])+2*(j[1]*e[2]-j[2]*e[1]))
    de2 = coeff1*((1-5*np.dot(j,n_p)**2)*(e[2]*j[0]-e[0]*j[2]) + 2*np.dot(j,n_p)*(e[2]*n_p[0]-e[0]*n_p[2])) + coeff2*(np.dot(j,ns)*(e[2]*ns[0]-e[0]*ns[2]) - 5*np.dot(e,ns)*(j[2]*ns[0]-j[0]*ns[2])+2*(j[2]*e[0]-j[0]*e[2]))
    de3 = coeff1*((1-5*np.dot(j,n_p)**2)*(e[0]*j[1]-e[1]*j[0]) + 2*np.dot(j,n_p)*(e[0]*n_p[1]-e[1]*n_p[0])) + coeff2*(np.dot(j,ns)*(e[0]*ns[1]-e[1]*ns[0]) - 5*np.dot(e,ns)*(j[0]*ns[1]-j[1]*ns[0])+2*(j[0]*e[1]-j[1]*e[0]))

    return [dj1,dj2,dj3,de1,de2,de3]

j0 = [0,0,1]
e0 = [0.002,0,0]
je0 = np.concatenate((j0,e0))

sol = nint.solve_ivp(ddt,(0, 3.14e10),je0)
print(sol)

j = sol['y'][:3]
e = sol['y'][3:]
t = sol["t"]

n = np.ones_like(j)
u = np.ones_like(e)

print(j)
print(e)

for i in range(len(j)):
    u[i] = e[i]/(np.sqrt(np.dot(e[i],e[i])))
    n[i] = j[i]/(np.sqrt(1-np.dot(e[i],e[i])**2))

magJ = np.sqrt(j[0]**2 + j[1]**2 + j[2]**2)

i = np.arccos(sol['y'][2])
i = np.rad2deg(i)


fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(n[0][-1000:], n[1][-1000:], n[2][-1000:])
ax.plot([0,n_p[0]],[0,n_p[1]],[0,n_p[2]])
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_zlim(-2,2)
ax.view_init(90,90)
fig.savefig("./ringMigration.png")



fig2, ax2 = plt.subplots(1,1)
ax2.plot(t[-1000:],n[0][-1000:], marker = "o", label ="x")
ax2.plot(t[-1000:],n[1][-1000:], marker = "o", label = "y")
ax2.plot(t[-1000:],n[2][-1000:], marker = "o", label = "z")
ax2.legend()

fig2.savefig("./ringMigration2D.png")







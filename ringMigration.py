import numpy as np
from scipy import integrate as nint
from matplotlib import pyplot as plt

G = 4*np.pi**2
M = 4.34e-5
J2 = 3343.43e-6
R = 1.6908e-4
a = 20*R
Ms = 1
ap = 19.165
n_p = [0,0,1]
n_s = [np.sin(np.deg2rad(97.77)),0,np.cos(np.deg2rad(97.77))]




def ddt(t, je):
    j = np.array(je[:3])
    e = np.array(je[3:])

    coeff1 = ((3*np.sqrt(G*M)*J2*R*R)/(2*a**(7/2)))
    coeff2 = 0#((3*np.sqrt(G)*Ms*a**(3/2))/(4*np.sqrt(M)*ap**3))

    

    dj = coeff1*np.dot(j,n_p)*np.cross(j,n_p) + coeff2*(np.dot(j,n_s)*np.cross(j,n_s)-5*np.dot(e,n_s)*np.cross(e,n_s))
    de = coeff1*((1-5*np.dot(j,n_p)**2)*np.cross(e,j) + 2*np.dot(j,n_p)*np.cross(e,n_p)) + coeff2*(np.dot(j,n_s)*(np.cross(e,n_s)) - 5*np.dot(e,n_s)*(np.cross(j,n_s))+2*(np.cross(j,e)))
    return [dj[0],dj[1],dj[2],de[0],de[1],de[2]]

#TODO: Figure out why e grows exponentially


eccen_0 = 0.002
inc_0 = np.deg2rad(97)
long_asc_node = np.deg2rad(0)
long_peri =np.deg2rad(0)
jhat = np.array([np.sin(inc_0)*np.cos(long_asc_node),np.sin(inc_0)*np.sin(long_asc_node),np.cos(inc_0)])
j0 = np.sqrt(1-eccen_0**2)*jhat
ehat = np.array([eccen_0*np.cos(long_asc_node),eccen_0*np.sin(long_asc_node),0])
e0 = ehat*np.cos(long_peri) + np.cross(jhat,ehat)*np.sin(long_peri) + jhat*np.dot(jhat,ehat)*(1-np.cos(long_peri))
je0 = np.concatenate((j0,e0))

nPoints = 1e4
time = np.linspace(0,1e5,nPoints)
sol = nint.odeint(ddt,je0,time)

print(sol)

'''
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
    ascendingNode = np.cross(n[i],[1,0,0])
    cosOmega = np.dot(np.cross(n[i],u[i]),[0,0,1])
    if np.abs(cosOmega) <= 1:
        if ascendingNode[1] > 0:
            Omega[i] = np.arccos(cosOmega)
        else:
            Omega[i] = 2*np.pi - np.arccos(cosOmega)
    elif cosOmega > 1:
        Omega[i] = 0
    else:
        Omega[i] = np.pi
    cosomega = np.dot(ascendingNode,u[i])
    if np.abs(cosomega) <= 1:
        if (np.dot(u[i],np.cross(ascendingNode,[0,0,1])) >= 0):
            omega[i] = np.arccos(cosomega)
        else: 
            omega[i] = 2*np.pi - np.arccos(cosomega)
    elif cosomega > 1:
        omega[i] = 0
    else:
        omega[i] = np.pi
    pomega[i] = (Omega[i] + omega[i]) % (2*np.pi)

magJ = np.sqrt(j[0]**2 + j[1]**2 + j[2]**2)
magE = np.sqrt(e[:,0]**2 + e[:,1]**2 + e[:,2]**2)

cosI = np.dot(n_p,np.transpose(n))
for i in range(len(cosI)):
    if cosI[i] > 1:
        cosI[i] = 1
    if cosI[i] < -1:
        cosI[i] = -1

inclination = np.arccos(cosI)


pomegaDotList = []
for i in range(1,len(pomega)):
    if np.abs(pomega[i-1] - pomega[i]) < 1.5*np.pi:
        pomegaDotList.append((pomega[i-1]-pomega[i])/(t[i-1]-t[i]))

pomegaDot = np.mean(pomegaDotList)
print(pomegaDot)




fig, ax = plt.subplots(2,1, figsize = (6,8))
ax[0].plot(t[-50:],Omega[-50:], marker = ".", label ="$\Omega$")
ax[0].plot(t[-50:],omega[-50:], marker = ".", label = "$\omega$")
ax[0].plot(t[-50:],pomega[-50:], marker = ".", label = "$\\varpi$")
ax[0].legend()
ax[0].set_xlabel("Years")
ax[0].set_ylabel("Angle")

ax[1].plot(t[0:],magE[0:], label = "|e|")
#ax[1].plot(t,inclination,label = "i")
ax[1].legend()
ax[1].set_xlabel("Years")
ax[1].set_ylabel("Eccentricity")


fig.savefig("./ringMigration2D.png")
'''







import numpy as np
from scipy import integrate as nint
from matplotlib import pyplot as plt

G = 4*np.pi**2 ##Units of Solar Masses, AU and yrs
M = 4.34e-5 ##Mass of Uranus in solar masses
J2 = 3343.43e-6 ##J2 of Uranus
R = 1.6908e-4 ##Radius of Uranus in AU
a = 20*R ##Semimajor axis of the test particle
Ms = 1 ##Mass of the sun in solar masses
ap = 19.165 ##Semi mahor axis of Uranus in AU
n_p = [0,0,1] ##Unit vector of Uranus's spin angular momentum
n_s = [np.sin(np.deg2rad(97.77)),0,np.cos(np.deg2rad(97.77))] ##Unit vector or Uranus's orbital angular momentum





def ddt(je, t):
    j = np.array(je[:3])
    e = np.array(je[3:])

    coeff1 = ((3*np.sqrt(G*M)*J2*R*R)/(2*a**(7/2))) ##J2 Contribution
    coeff2 = 0#((3*np.sqrt(G)*Ms*a**(3/2))/(4*np.sqrt(M)*ap**3)) ##Solar Contribution

    

    dj = coeff1*np.dot(j,n_p)*np.cross(j,n_p) + coeff2*(np.dot(j,n_s)*np.cross(j,n_s)-5*np.dot(e,n_s)*np.cross(e,n_s))
    de = coeff1*((1-5*np.dot(j,n_p)**2)*np.cross(e,j) + 2*np.dot(j,n_p)*np.cross(e,n_p)) + coeff2*(np.dot(j,n_s)*(np.cross(e,n_s)) +2*(np.cross(j,e)) - 5*np.dot(e,n_s)*(np.cross(j,n_s)))

    return [dj[0],dj[1],dj[2],de[0],de[1],de[2]]

#TODO: Figure out why e grows exponentially


##Initial Parameters to define
eccen_0 = 0.002 ##Eccentricity
inc_0 = np.deg2rad(0)##Inclination
long_asc_node = np.deg2rad(0)##Longitude of Ascending Node (Omega)
long_peri =np.deg2rad(0)##Longitude of Pericenter (omega)

##Defining j (~anglular momentum) and e(~Runge-Lenz Vector)
jhat = np.array([np.sin(inc_0)*np.cos(long_asc_node),np.sin(inc_0)*np.sin(long_asc_node),np.cos(inc_0)])
j0 = np.sqrt(1-eccen_0**2)*jhat
ehat = np.array([eccen_0*np.cos(long_asc_node),eccen_0*np.sin(long_asc_node),0])
e0 = ehat*np.cos(long_peri) + np.cross(jhat,ehat)*np.sin(long_peri) + jhat*np.dot(jhat,ehat)*(1-np.cos(long_peri))
je0 = np.concatenate((j0,e0))

nPoints = int(1e3)
time = np.linspace(0,1e5,nPoints)
sol = nint.odeint(ddt,je0,time)



j = sol[:,:3]
e = sol[:,3:]
t = time

n = np.ones_like(j)
u = np.ones_like(e)
Omega = np.ones(len(n))
omega = np.ones(len(n))
pomega = np.ones(len(n))


##Converting j and e to Euler elements
for i in range(len(j)):
    u[i] = e[i]/(np.sqrt(np.dot(e[i],e[i]))) if np.dot(e[i],e[i]) != 0 else [0,0,0] ##Turning e into a unit vector
    n[i] = j[i]/(np.sqrt(1-np.dot(e[i],e[i]))) ##Turning j into a unit vector
    ascendingNode = np.cross(n[i],[0,0,1]) if  np.linalg.norm(np.cross(n[i],[0,0,1])) != 0 else [1,0,0]
    cosOmega = np.dot(ascendingNode,[1,0,0])
    if np.abs(cosOmega) <= 1: ##Sometimes abs(cosOmega) is a little bit bigger than 1 (I checked that its only a little)
        if ascendingNode[1] >= 0: ##Reference direction in [1,0,0] so 
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


magE = np.sqrt(e[:,0]**2 + e[:,1]**2 + e[:,2]**2)

##I think this is wrong I have to check
cosI = np.dot(n_p,np.transpose(n))
for i in range(len(cosI)):
    if cosI[i] > 1:
        cosI[i] = 1
    if cosI[i] < -1:
        cosI[i] = -1

inclination = np.arccos(cosI)

##Calculating pomega dot
pomegaDotList = []
for i in range(1,len(pomega)):
    if np.abs(pomega[i-1] - pomega[i]) < 1.7*np.pi: ##Adjusting when it wraps around 2pi
        pomegaDotList.append((pomega[i-1]-pomega[i])/(t[i-1]-t[i]))
    elif pomega[i-1] - pomega[i] < 0:
        pomegaDotList.append((pomega[i-1]-pomega[i] + 2*np.pi)/(t[i-1]-t[i]))
    else:
        pomegaDotList.append((pomega[i-1]-pomega[i] - 2*np.pi)/(t[i-1]-t[i]))

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








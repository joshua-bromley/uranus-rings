import numpy as np
from scipy import integrate as nint
from matplotlib import pyplot as plt

G = 4*np.pi**2 ##Units of Solar Masses, AU and yrs
M = 4.34e-5 ##Mass of Uranus in solar masses
J2 = 3343.43e-6 ##J2 of Uranus
R = 1.6908e-4 ##Radius of Uranus in AU
a = 100*R ##Semimajor axis of the test particle
Ms = 1 ##Mass of the sun in solar masses
ap = 19.165 ##Semi major axis of Uranus in AU
n_p = [0,0,1] ##Unit vector of Uranus's spin angular momentum
n_s = [np.sin(np.deg2rad(97.77)),0,np.cos(np.deg2rad(97.77))] ##Unit vector or Uranus's orbital angular momentum


##Initial Parameters to define
eccen_0 = 0.01 ##Eccentricity
inc_0 = np.deg2rad(100)##Inclination
long_asc_node = np.deg2rad(0)##Longitude of Ascending Node (Omega)
long_peri =np.deg2rad(90)##Longitude of Pericenter (omega)

##Defining j (~anglular momentum) and e(~Runge-Lenz Vector)
jhat = np.array([np.sin(inc_0)*np.cos(long_asc_node),np.sin(inc_0)*np.sin(long_asc_node),np.cos(inc_0)])
j0 = np.sqrt(1-eccen_0**2)*jhat
ehat = np.array([eccen_0*np.cos(long_asc_node),eccen_0*np.sin(long_asc_node),0])
e0 = ehat*np.cos(long_peri) + np.cross(jhat,ehat)*np.sin(long_peri) + jhat*np.dot(jhat,ehat)*(1-np.cos(long_peri))
je0 = np.concatenate((j0,e0))

omJ2 = ((3*np.sqrt(G*M)*J2*R*R)/(2*a**(7/2)*(1-eccen_0**2)**(5/2)))
omS = ((3*np.sqrt(G*M)*Ms*a**(3/2))/(4*M*ap**3))*np.dot(j0,n_s)
T = np.sqrt(4*np.pi**2/(G*M)*a**3)
T_omJ2 = 2*np.pi/(omJ2)
T_omS = 2*np.pi/(omS)
print(f"Orbital Period = {T} yrs = {365.25*T} days")
print(f"J2 Procession Period = {T_omJ2} yrs")
print(f"Solar Procession Period = {T_omS} yrs")
tend = 100*T_omJ2 if T_omJ2 < T_omS else 100*T_omS



def ddt(je, t):
    j = np.array(je[:3])
    e = np.array(je[3:])
    eccen = np.sqrt(np.dot(e,e))

    coeff1 = ((3*np.sqrt(G*M)*J2*R*R)/(2*a**(7/2)*(1-eccen**2)**(5/2))) ##J2 Contribution
    coeff2 = ((3*np.sqrt(G)*Ms*a**(3/2))/(4*np.sqrt(M)*ap**3)) ##Solar Contribution

    

    dj = coeff1*np.dot(j,n_p)*np.cross(j,n_p) + coeff2*(np.dot(j,n_s)*np.cross(j,n_s)-5*np.dot(e,n_s)*np.cross(e,n_s))
    de = 0.5*coeff1*((1-(5*np.dot(j,n_p)**2)/(1-eccen**2))*np.cross(e,j) + 2*np.dot(j,n_p)*np.cross(e,n_p)) + coeff2*(np.dot(j,n_s)*(np.cross(e,n_s)) +2*(np.cross(j,e)) - 5*np.dot(e,n_s)*(np.cross(j,n_s)))

    return [dj[0],dj[1],dj[2],de[0],de[1],de[2]]





nPoints = int(1e4)
time = np.linspace(0,tend,nPoints)
sol = nint.odeint(ddt,je0,time)



j = sol[:,:3]
e = sol[:,3:]
t = time

n = np.ones_like(j)
u = np.ones_like(e)
Omega = np.ones(len(n))
omega = np.ones(len(n))
pomega = np.ones(len(n))
fakePomega = np.ones(len(n))


##Converting j and e to Euler elements
for i in range(len(j)):
    u[i] = e[i]/(np.sqrt(np.dot(e[i],e[i]))) if np.dot(e[i],e[i]) != 0 else [0,0,0] ##Turning e into a unit vector
    n[i] = j[i]/(np.sqrt(1-np.dot(e[i],e[i]))) ##Turning j into a unit vector
    ascendingNode = np.cross(n[i],[0,0,1]) if  np.linalg.norm(np.cross(n[i],[0,0,1])) != 0 else [1,0,0]
    ascendingNode = ascendingNode/(np.sqrt(np.dot(ascendingNode,ascendingNode)))
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
    sinomega = np.dot(n[i],np.cross(ascendingNode,u[i]))
    if np.abs(cosomega) <= 1:
        #if (np.dot(n[i],np.cross(ascendingNode,u[i])) >= 0):
        #    omega[i] = np.arccos(cosomega)
        #else: 
        #    omega[i] = 2*np.pi - np.arccos(cosomega)
        omega[i] = np.arctan2(sinomega,cosomega) if np.arctan2(sinomega,cosomega) >= 0 else 2*np.pi + np.arctan2(sinomega,cosomega)
    elif cosomega > 1:
        omega[i] = 0
    else:
        omega[i] = np.pi
    pomega[i] = (Omega[i] + omega[i]) % (2*np.pi)
    cosFakePomega = np.dot(u[i],[1,0,0])
    if np.abs(cosFakePomega) <= 1:
        if u[i][1] >= 0:
            fakePomega[i] = np.arccos(cosFakePomega)
        else:
            fakePomega[i] = 2*np.pi - np.arccos(cosFakePomega)
    elif cosFakePomega > 1:
        fakePomega[i] = 0
    else:
        fakePomega[i] = np.pi


magE = np.sqrt(e[:,0]**2 + e[:,1]**2 + e[:,2]**2)


##I think this is wrong I have to check
cosI = np.dot(n_p,np.transpose(n))
for i in range(len(cosI)):
    if cosI[i] > 1:
        cosI[i] = 1
    if cosI[i] < -1:
        cosI[i] = -1

inclination = np.rad2deg(np.arccos(cosI))
avgN = np.mean(n, axis = 0)
avgN = avgN/(np.sqrt(np.dot(avgN,avgN)))
avgInclination = np.rad2deg(np.arccos(np.dot(n_p,avgN)))
print(avgInclination)
print(np.std(inclination-avgInclination))

##Calculating pomega dot
omegaDotList = []
OmegaDotList = []
pomegaDotList = []
fakePomegaDotList = []
for i in range(1,len(pomega)):
    ##Pomega
    if np.abs(pomega[i-1] - pomega[i]) < 1.8*np.pi: ##Adjusting when it wraps around 2pi
        pomegaDotList.append((pomega[i-1]-pomega[i])/(t[i-1]-t[i]))
    elif pomega[i-1] - pomega[i] > 0:
        #pomegaDotList.append((pomega[i-1]-pomega[i] + 2*np.pi)/(t[i-1]-t[i]))
        hi = "hi"
    else:
        #pomegaDotList.append((pomega[i-1]-pomega[i] - 2*np.pi)/(t[i-1]-t[i]))
        hi = "hi"
    ##omega
    if np.abs(omega[i-1] - omega[i]) < 1.8*np.pi: ##Adjusting when it wraps around 2pi
        omegaDotList.append((omega[i-1]-omega[i])/(t[i-1]-t[i]))
    elif omega[i-1] - omega[i] > 0:
        #omegaDotList.append((omega[i-1]-omega[i] + 2*np.pi)/(t[i-1]-t[i]))
        hi = "hi"
    else:
        #omegaDotList.append((omega[i-1]-omega[i] - 2*np.pi)/(t[i-1]-t[i]))
        hi ="hi"
    ##Omega
    if np.abs(Omega[i-1] - Omega[i]) < 1.8*np.pi: ##Adjusting when it wraps around 2pi
        OmegaDotList.append((Omega[i-1]-Omega[i])/(t[i-1]-t[i]))
    elif Omega[i-1] - Omega[i] > 0:
        #OmegaDotList.append((Omega[i-1]-Omega[i] + 2*np.pi)/(t[i-1]-t[i]))
        hi = "hi"
    else:
        #OmegaDotList.append((Omega[i-1]-Omega[i] - 2*np.pi)/(t[i-1]-t[i]))
        hi = "hi"
    if np.abs(fakePomega[i-1] - fakePomega[i]) < 1.8*np.pi: ##Adjusting when it wraps around 2pi
        fakePomegaDotList.append((fakePomega[i-1]-fakePomega[i])/(t[i-1]-t[i]))
    elif fakePomega[i-1] - fakePomega[i] > 0:
        #pomegaDotList.append((pomega[i-1]-pomega[i] + 2*np.pi)/(t[i-1]-t[i]))
        hi = "hi"
    else:
        #pomegaDotList.append((pomega[i-1]-pomega[i] - 2*np.pi)/(t[i-1]-t[i]))
        hi = "hi"

pomegaDot = np.mean(pomegaDotList)
omegaDot = np.mean(omegaDotList)
OmegaDot = np.mean(OmegaDotList)
fakePomegaDot = np.mean(fakePomegaDotList)

omegaDotTheory = ((np.sqrt(G*M)*J2*R*R)/(4*a**(7/2)*(1-eccen_0**2)**2)) * (15*np.cos(inc_0)**2 - 3)
OmegaDotTheory = ((-3*np.sqrt(G*M)*J2*R*R*np.cos(inc_0))/(2*a**(7/2)*(1-eccen_0**2)**2))
pomegaDotTheory = omegaDotTheory + OmegaDotTheory

print(f"Pomega Dot = {pomegaDot} rad/yr by simulation and {pomegaDotTheory} rad/yr by theory")
print(f"The difference between the pomegas is {pomegaDot-pomegaDotTheory}")
#print(f"~Pomega Dot = {fakePomegaDot} rad/yr")
print(f"omega Dot = {omegaDot} rad/yr by simulation and {omegaDotTheory} rad/yr by theory")
print(f"The difference between the omegas is {omegaDot-omegaDotTheory}")
print(f"Omega Dot = {OmegaDot} rad/yr by simulation and {OmegaDotTheory} rad/yr by theory")
print(f"The difference between the Omegas is {OmegaDot-OmegaDotTheory}")


t = t/T_omJ2 if T_omJ2 < T_omS else t/T_omS


fig, ax = plt.subplots(3,1, figsize = (6,12))
ax[0].plot(t[-700:],Omega[-700:], marker = ".", label ="$\Omega$")
ax[0].plot(t[-700:],omega[-700:], marker = ".", label = "$\omega$")
ax[0].plot(t[-700:],pomega[-700:], marker = ".", label = "$\\varpi$")
#ax[0].plot(t[-700:],fakePomega[-700:], marker = ".", label = "Fake $\\varpi$")
ax[0].legend()
ax[0].set_xlabel("Precession Periods")
ax[0].set_ylabel("Angle")

ax[1].plot(t[0:],magE[0:], label = "|e|")

ax[1].set_ylim(0,1)
ax[1].legend()
ax[1].set_xlabel("Preciession Periods")
ax[1].set_ylabel("Eccentricity")

ax[2].plot(t,inclination,label = "i")
ax[2].set_ylim(0,180)
ax[2].legend()
ax[2].set_xlabel("Precession Periods")
ax[2].set_ylabel("Inclination")


fig.savefig("./ringMigration2D.png")

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection="3d")
ax2.plot(n[:,0],n[:,1],n[:,2])
ax2.plot((0,avgN[0]),(0,avgN[1]),(0,avgN[2]))
ax2.set_xlim(-1.1,1.1)
ax2.set_ylim(-1.1,1.1)
ax2.set_zlim(-1.1,1.1)
#ax2.view_init(90,0)

plt.show()








import numpy as np
from scipy import integrate as nint
from matplotlib import pyplot as plt
from tqdm import tqdm

G = 4*np.pi**2 ##Units of Solar Masses, AU and yrs
M = 4.34e-5 ##Mass of Uranus in solar masses
J2 = 3343.43e-6 ##J2 of Uranus
R = 1.6908e-4 ##Radius of Uranus in AU
Ms = 1 ##Mass of the sun in solar masses
ap = 19.165 ##Semi major axis of Uranus in AU
n_p = [0,0,1] ##Unit vector of Uranus's spin angular momentum
n_s = [np.sin(np.deg2rad(30)),0,np.cos(np.deg2rad(30))] ##Unit vector or Uranus's orbital angular momentum

aList = np.logspace(np.log10(2*R),np.log10(4000*R)) ##Semi-maxot axes to try

def ddt(je, t):
    j = np.array(je[:3])
    e = np.array(je[3:])
    eccen = np.sqrt(np.dot(e,e)) 

    coeff1 = ((3*np.sqrt(G*M)*J2*R*R)/(2*a**(7/2)*(1-eccen**2)**(5/2))) ##J2 Contribution
    coeff2 = ((3*np.sqrt(G)*Ms*a**(3/2))/(4*np.sqrt(M)*ap**3)) ##Solar Contribution

    

    dj = coeff1*np.dot(j,n_p)*np.cross(j,n_p) + coeff2*(np.dot(j,n_s)*np.cross(j,n_s)-5*np.dot(e,n_s)*np.cross(e,n_s))
    de = 0.5*coeff1*((1-(5*np.dot(j,n_p)**2)/(1-eccen**2))*np.cross(e,j) + 2*np.dot(j,n_p)*np.cross(e,n_p)) + coeff2*(np.dot(j,n_s)*(np.cross(e,n_s)) +2*(np.cross(j,e)) - 5*np.dot(e,n_s)*(np.cross(j,n_s)))

    return [dj[0],dj[1],dj[2],de[0],de[1],de[2]]

def ringEvolution(a, eccen0 = 0.01, inc0 = 0, longAscNode0 = 0, longPeri0 = 0, T = 1e3, N = 1e4):

    ##Defining j (~anglular momentum) and e(~Runge-Lenz Vector)
    jhat = np.array([np.sin(inc0)*np.cos(longAscNode0),np.sin(inc0)*np.sin(longAscNode0),np.cos(inc0)])
    j0 = np.sqrt(1-eccen0**2)*jhat
    ehat = np.array([eccen0*np.cos(longAscNode0),eccen0*np.sin(longAscNode0),0])
    e0 = ehat*np.cos(longPeri0) + np.cross(jhat,ehat)*np.sin(longPeri0) + jhat*np.dot(jhat,ehat)*(1-np.cos(longPeri0))
    je0 = np.concatenate((j0,e0))

    omJ2 = ((3*np.sqrt(G*M)*J2*R*R)/(2*a**(7/2)*(1-eccen0**2)**(5/2)))
    omS = ((3*np.sqrt(G*M)*Ms*a**(3/2))/(4*M*ap**3))*np.dot(j0,n_s)
    T_omJ2 = 2*np.pi/(omJ2)
    T_omS = 2*np.pi/(omS)
    tEnd = T*T_omJ2 if T_omJ2 < T_omS else T*T_omS

    time = np.linspace(0,tEnd,int(N))
    sol = nint.odeint(ddt,je0,time)

    j = sol[:,:3]
    e = sol[:,3:]
    t = time

    n = np.ones_like(j)
    u = np.ones_like(e)

    for i in range(len(j)):
        u[i] = e[i]/(np.sqrt(np.dot(e[i],e[i]))) if np.dot(e[i],e[i]) != 0 else [0,0,0] ##Turning e into a unit vector
        n[i] = j[i]/(np.sqrt(1-np.dot(e[i],e[i]))) ##Turning j into a unit vector


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
    spread = np.std(inclination-avgInclination)
    return avgInclination, spread

iList = []
spreadList = []

for a in tqdm(aList):
    testIncs = [0,15,30]
    results = []
    for I in testIncs:
        results.append(ringEvolution(a,inc0 = I, T = 100, N = 1e3))
    results = np.asarray(results)
    mostConverged = np.argmin(results[:,1])
    iList.append(results[mostConverged,0])
    spreadList.append(results[mostConverged,1])

spreadList = np.asarray(spreadList)
spreadList = 5 * spreadList / np.max(spreadList)

aList = aList/R

fig, ax = plt.subplots(1,1)
ax.plot(aList,iList, ls = "", marker = ".")
ax.set_xscale("log")
ax.set_ylabel("Inclination (Degrees)")
ax.set_xlabel("Semi Major Axis $(R_p)$")
fig.savefig("./laplacePlane.png")
plt.show()





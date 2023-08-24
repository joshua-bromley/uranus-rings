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
inc_s = 97.77
e_s = 0.04717
rL = (J2*R*R*(ap**3)*((1-e_s**2)**(3/2))*M/Ms)**(1/5)
n_s = [np.sin(np.deg2rad(inc_s)),0,np.cos(np.deg2rad(inc_s))] ##Unit vector or Uranus's orbital angular momentum
print(rL/R)
aList = np.linspace(100*R,1000*R,100) ##Semi-major axes to try

def ddt(je, t):
    j = np.array(je[:3])
    e = np.array(je[3:])
    eccen = np.sqrt(np.dot(e,e))

    coeff1 = ((3*np.sqrt(G*M)*J2*R*R)/(2*a**(7/2)*(1-eccen**2)**(5/2))) ##J2 Contribution
    coeff2 = ((3*np.sqrt(G)*Ms*a**(3/2))/(4*np.sqrt(M)*ap**3)) ##Solar Contribution

    

    dj = coeff1*np.dot(j,n_p)*np.cross(j,n_p) + coeff2*(np.dot(j,n_s)*np.cross(j,n_s)-5*np.dot(e,n_s)*np.cross(e,n_s))
    de = 0.5*coeff1*((1-(5*np.dot(j,n_p)**2)/(1-eccen**2))*np.cross(e,j) + 2*np.dot(j,n_p)*np.cross(e,n_p)) + coeff2*(np.dot(j,n_s)*(np.cross(e,n_s)) +2*(np.cross(j,e)) - 5*np.dot(e,n_s)*(np.cross(j,n_s)))

    return [dj[0],dj[1],dj[2],de[0],de[1],de[2]]

def ringEvolution(a, eccen_0 = 0.01, inc_0 = 0, long_asc_node = 0, long_peri = 0, T = 1e3, N = 1e4):

    inc_0 = np.deg2rad(inc_0)
    long_asc_node = np.deg2rad(long_asc_node)
    long_peri = np.deg2rad(long_peri)
    ##Defining j (~anglular momentum) and e(~Runge-Lenz Vector)
    jhat = np.array([np.sin(inc_0)*np.cos(long_asc_node),np.sin(inc_0)*np.sin(long_asc_node),np.cos(inc_0)])
    j0 = np.sqrt(1-eccen_0**2)*jhat
    ehat = np.array([eccen_0*np.sin(long_asc_node),eccen_0*np.cos(long_asc_node),0])
    e0 = ehat*np.cos(long_peri) + np.cross(jhat,ehat)*np.sin(long_peri) + jhat*np.dot(jhat,ehat)*(1-np.cos(long_peri))
    je0 = np.concatenate((j0,e0))


    omJ2 = ((3*np.sqrt(G*M)*J2*R*R)/(2*a**(7/2)*(1-eccen_0**2)**(5/2)))
    omS = ((3*np.sqrt(G*M)*Ms*a**(3/2))/(4*M*ap**3))*np.dot(j0,n_s)
    #T = np.sqrt(4*np.pi**2/(G*M)*a**3)
    T_omJ2 = 2*np.pi/(omJ2)
    T_omS = 2*np.pi/(omS)
    tend = T#T*T_omJ2 if T_omJ2 < T_omS else T*T_omS

    nPoints = int(N)
    time = np.linspace(0,tend,nPoints)
    sol = nint.odeint(ddt,je0,time)

    j = sol[:,:3]
    e = sol[:,3:]
    t = time

    return j,e,t 

eList = []
jList = []
time = []
e0List = []

for a in aList:
    e0 = 1 - 2*R/a
    e0List.append(e0)
    j,e,t = ringEvolution(a = a, eccen_0=e0, inc_0= 97.77, T = 1e5, N = 1e6)
    eList.append(e)
    jList.append(j)
    time = t

eList = np.asarray(eList)
print(eList.shape)



fig2 = plt.figure()
ax2 = fig2.add_subplot(projection="3d")
for i in range(len(eList)):
    ax2.plot(eList[i,:,0],eList[i,:,1],eList[i,:,2])

ax2.set_xlim(-1.1,1.1)
ax2.set_ylim(-1.1,1.1)
ax2.set_zlim(-1.1,1.1)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")
fig2.savefig("./diskFormation.png")

plt.show()

finalJ = []
finalE = []
for i in range(len(aList)):
    finalJ.append(jList[i][-1])
    finalE.append(eList[i][-1])

finalJ = np.asarray(finalJ)
finalE = np.asarray(finalE)

disks = [] 
for i in range(len(aList)):
    disk = []
    jNorm = finalJ[i]/np.sqrt(np.dot(finalJ[i],finalJ[i]))
    e = np.sqrt(np.dot(finalE[i],finalE[i]))
    periVec = finalE[i]/e
    angles = np.linspace(0,2*np.pi,100)
    a = aList[i]
    for theta in angles:
        r = a*(1-e*e)/(1+e*np.cos(theta))
        x = periVec*r
        x = x*np.cos(theta) + np.cross(jNorm,x)*np.sin(theta) + jNorm*np.dot(jNorm,x)*(1-np.cos(theta))
        disk.append(x)

    disk = np.asarray(disk) 
    disks.append(disk)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
for i in range(len(disks)):
    ax.plot(disks[i][:,0],disks[i][:,1],disks[i][:,2])

fig.savefig("./disk.png")

fig3, ax3 = plt.subplots(1,1)
for i in range(len(disks)):
    for j in range(len(disks[i])):
        if (np.sqrt(disks[i][j,0]**2 + disks[i][j,1]**2 + disks[i][j,2]**2) < 3*R):
            ax3.plot(disks[i][j,0],disks[i][j,2],alpha = 0.8,color = "tab:blue",marker = ",")
        #print(disks[i][j,0],disks[i][j,2])
ax3.grid(axis = "x", color = '0.95')
ax3.grid(axis = "y", color = "0.95")

fig3.savefig("./diskCrossSection.png")



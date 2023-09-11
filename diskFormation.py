import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from scipy import integrate as nint
from tqdm import tqdm
from matplotlib.animation import FFMpegWriter

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
aList = np.linspace(100*R,1000*R,5) ##Semi-major axes to try

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
    j,e,t = ringEvolution(a = a, eccen_0=e0, inc_0= 0, T = 2e4, N = 2e5)
    eList.append(e)
    jList.append(j)
    time = t

eList = np.asarray(eList)
jList = np.asarray(jList)
print(eList.shape)

eList = eList[:,::500,:]
jList = jList[:,::500,:]
time = time[::500]


eListX = eList[:,:,0]
eListY = eList[:,:,1]
eListZ = eList[:,:,2]

magEList = np.ones((len(eList),len(eList[0])))

n = np.ones_like(jList)
u = np.ones_like(eList)
omega = np.ones((len(eList),len(eList[0])))

for i in (range(len(eList))):
    for j in tqdm(range(len(eList[i]))):
        magEList[i][j] = np.sqrt(np.dot(eList[i,j],eList[i,j]))
        n[i,j] = jList[i,j]/(np.sqrt(np.dot(jList[i,j],jList[i,j])))
        u[i,j] = eList[i,j]/np.sqrt(np.dot(eList[i,j],eList[i,j]))

        ascendingNode = np.cross(n[i,j],[0,0,1]) if  np.linalg.norm(np.cross(n[i,j],[0,0,1])) != 0 else [1,0,0]
        ascendingNode = ascendingNode/(np.sqrt(np.dot(ascendingNode,ascendingNode)))
        cosomega = np.dot(ascendingNode,u[i,j])
        sinomega = np.dot(n[i,j],np.cross(ascendingNode,u[i,j]))
        if np.abs(cosomega) <= 1:
            omega[i,j] = np.arctan2(sinomega,cosomega) if np.arctan2(sinomega,cosomega) >= 0 else 2*np.pi + np.arctan2(sinomega,cosomega)
        elif cosomega > 1:
            omega[i,j] = 0
        else:
            omega[i,j] = np.pi


cosI = np.ones((len(n),len(n[0]))) 
for i in range(len(n)):
    cosI[i] = np.dot(n_p,np.transpose(n[i]))
    for j in range(len(cosI)):
        if cosI[i][j] > 1:
            cosI[i][j] = 1
        if cosI[i][j] < -1:
            cosI[i][j] = -1

inclination = np.rad2deg(np.arccos(cosI))

colors = ["tab:blue","tab:orange", "tab:green", "tab:red", "tab:purple"]



def update(t):
    ax.cla()
    ax1.cla()
    ax2.cla()
    ax3.cla()

    for i in range(len(eListX)):
        x = eListX[i,t]
        y = eListY[i,t]
        z = eListZ[i,t]

        ax.plot([0,x],[0,y],[0,z], color = colors[i])
        ax1.plot(time[:t],magEList[i,:t], color = colors[i],label = "{0:.1f} $R_p$".format(aList[i]/R))
        ax2.plot(time[:t], inclination[i,:t], color = colors[i])
        ax3.plot(time[:t], omega[i,:t], color = colors[i])
    ax.plot([0,0],[0,0],[0,1], color = "black")
    ax1.legend(loc = "lower left",frameon = False)
    


    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ax1.set_ylim(0,1)
    ax1.set_ylabel("Eccentricity")
    ax1.set_xlabel("Time (yrs)")

    ax2.set_ylim(0,180)
    ax2.set_ylabel("Inclination (Degrees)")
    ax2.set_xlabel("Time (yrs)")

    ax3.set_ylim(0,2*np.pi)
    ax3.set_ylabel("omega (Radians)")
    ax3.set_xlabel("Time (yrs)")


fig = plt.figure(dpi=100, figsize = (10,8))
ax = fig.add_subplot(2,2,1,projection='3d')
ax1 = fig.add_subplot(2,2,2)
ax2 = fig.add_subplot(2,2,3)
ax3 = fig.add_subplot(2,2,4)

ani = animation.FuncAnimation(fig = fig, func = update, frames = 400, interval = 30)
writer = animation.FFMpegWriter(fps = 50)

'''
with writer.saving(fig, "./diskEvolution_300_500.mp4", dpi = 100):
    for t in range(0,400):
        ax.cla()
        ax1.cla()
        ax2.cla()
        ax3.cla()

        for i in range(len(eListX)):
            x = eListX[i,t]
            y = eListY[i,t]
            z = eListZ[i,t]

            ax.plot([0,x],[0,y],[0,z], color = colors[i])
            ax1.plot(time[:t],magEList[i,:t], color = colors[i],label = "{0:.1f} $R_p$".format(aList[i]/R))
            ax2.plot(time[:t], inclination[i,:t], color = colors[i])
            ax3.plot(time[:t], omega[i,:t], color = colors[i])
        ax.plot([0,0],[0,0],[0,1], color = "black")
        ax1.legend(loc = "lower left",frameon = False)
        


        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        ax1.set_ylim(0,1)
        ax1.set_ylabel("Eccentricity")
        ax1.set_xlabel("Time (yrs)")

        ax2.set_ylim(0,180)
        ax2.set_ylabel("Inclination (Degrees)")
        ax2.set_xlabel("Time (yrs)")

        ax3.set_ylim(0,2*np.pi)
        ax3.set_ylabel("omega (Radians)")
        ax3.set_xlabel("Time (yrs)")
        plt.draw()
        plt.pause(0.1)
        writer.grab_frame()
'''
#ani.save("diskEvolution_{0:.0f}_{1:.0f}.gif".format(aList[0]/R,aList[-1]/R), writer = writer)

plt.show()     












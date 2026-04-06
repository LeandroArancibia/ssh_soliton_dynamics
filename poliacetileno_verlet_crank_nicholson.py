#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%matplotlib inline
get_ipython().run_line_magic('matplotlib', 'notebook')
from IPython.display import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.sparse import diags
import scipy.optimize as opt
from time import time
#from sympy import *


# In[2]:


# definir constantes 
deltaTiempo = 0.01
tiempoTotal = 35
NPasos = int(tiempoTotal / deltaTiempo)
x_tiempo = np.linspace(0,tiempoTotal,NPasos)
tOff = 35
campoElectrico = 0.02
NSitios = 99
espinesUp = 50 # si es impar colocar el mas grande aca
espinesDown = 50
t0hBarraOmegaQ = 1 / ( 6.582 * 0.01 ) # t0 / (hbar * omegaQ)
kaSobret0 = 21 * np.square(1.22) / 2.5
gamma = 2.0008
beta = 4.1 / (1.22*21)
tolerancia = 1e-7
omegaQ = 2.5e14 


# In[3]:


error = 1.0
yn = np.random.uniform(-0.1,0.1,NSitios)
ynAux = np.zeros(NSitios)
j=0
while error > tolerancia:
    diagonalAuxiliar = yn * gamma - 1 
    diagonales = [diagonalAuxiliar[0:-1] , np.conj(diagonalAuxiliar)[0:-1]]
    hamiltoniano = diags(diagonales,[-1,1]).toarray()
    hamiltoniano[0,-1] = diagonalAuxiliar[-1]
    hamiltoniano[-1,0] = np.conj(diagonalAuxiliar[-1])
    autoValores,autoVectores = np.linalg.eig(hamiltoniano)
    idx = autoValores.argsort()[::1]   
    autoValores = autoValores[idx]
    autoVectores = autoVectores[:,idx]
    for n in range(NSitios):
        ynAux[n] =  np.vdot(np.roll(autoVectores,-1, axis=0)[n,0:espinesUp],autoVectores[n,0:espinesUp])
        ynAux[n] += np.vdot(np.roll(autoVectores,-1, axis=0)[n,0:espinesDown],autoVectores[n,0:espinesDown])
    ynAux *= -2 * beta
    ynAux -= sum(ynAux) / NSitios
    error = np.sqrt( sum( np.square(ynAux - yn) ) / NSitios )
    yn[:] = ynAux[:]
    j +=1 


# In[4]:


def yn_punto_punto(ienes,psiUp,psiDown,tiempo):
    aceleracion = np.array(np.diag( np.conj( np.roll(psiUp,-1, axis=0) ) @ np.transpose( np.roll(psiUp,-2, axis=0) ) ))
    aceleracion -= np.array( 2 * np.diag( np.conj( psiUp ) @ np.transpose( np.roll(psiUp,-1, axis=0) ) ) )
    aceleracion += np.array( np.diag( np.conj( np.roll(psiUp,+1, axis=0) ) @ np.transpose( psiUp ) ) )
    aceleracion += np.array( np.diag( np.conj( np.roll(psiDown,-1, axis=0) ) @ np.transpose( np.roll(psiDown,-2, axis=0) ) ) )
    aceleracion -= np.array( 2 * np.diag( np.conj( psiDown ) @ np.transpose( np.roll(psiDown,-1, axis=0) ) ) )
    aceleracion += np.array( np.diag( np.conj( np.roll(psiDown,+1, axis=0) ) @ np.transpose( psiDown ) ) )
    aceleracion *= np.exp(-1j * campoElectrico * tiempo) * beta / 4
    #aceleracion *= np.exp(-1j * campoElectrico / omegaQ *  np.tanh(tiempo * 0.25) ) * beta / 4
    aceleracion += np.array( np.conj(aceleracion)    )
    aceleracion -= np.array( 1 / 4 * (2*ienes - np.roll(ienes,-1) - np.roll(ienes,1)) )
    #print("aceleracion= ",aceleracion[0].real)
    return aceleracion.real

def matriz_U(ienes,tiempo):
    diagonalAuxiliar = (ienes * gamma - 1) *  np.exp(-1j * campoElectrico * tiempo)
    #diagonalAuxiliar = (ienes * gamma - 1) *  np.exp(-1j * campoElectrico / omegaQ * np.tanh(tiempo * 0.25)  )
    diagonales = [diagonalAuxiliar[0:-1] , np.conj(diagonalAuxiliar)[0:-1]]
    hamiltoniano = diags(diagonales,[-1,1]).toarray()
    hamiltoniano[0,-1] = diagonalAuxiliar[-1]
    hamiltoniano[-1,0] = np.conj(diagonalAuxiliar[-1])
    #construir U
    numerador = np.identity(NSitios) - 1.0j * t0hBarraOmegaQ * deltaTiempo / 2.0 * hamiltoniano
    denominador = np.identity(NSitios) + 1.0j * t0hBarraOmegaQ * deltaTiempo / 2.0 * hamiltoniano
    U = np.dot(numerador,np.linalg.inv(denominador))
    return U


# In[12]:


# crear variables

ynActual = np.zeros(NSitios)
ynNuevo = np.zeros((NPasos,NSitios),dtype=float)
velocidadActual = np.zeros(NSitios)
velocidadNueva = np.zeros_like(ynNuevo)
psiActual = np.zeros((NSitios,NSitios),dtype=complex)
psiNuevo = np.zeros((NSitios,NSitios),dtype=complex)
phiPorEpsilon = np.zeros((NSitios,NSitios),dtype=complex)
rhoVector = np.zeros_like(ynNuevo)
aceleracionActual = np.zeros(NSitios)
aceleracionNueva = np.zeros(NSitios)
# variables a medir
energiaElectrones = np.zeros(NPasos,dtype=float)
energiaElectronesAuxiliar = np.zeros(NSitios,dtype=float)
energiaIones = np.zeros(NPasos,dtype=float)
energiaPotencial = np.zeros(NPasos,dtype=float)
energiaCinetica = np.zeros(NPasos,dtype=float)
energiaIonesAuxiliar = np.zeros(NSitios,dtype=float)
energiaPotencialAuxiliar = np.zeros((NSitios),dtype=float)
energiaCineticaAuxiliar = np.zeros((NSitios),dtype=float)
# asignar condiciones iniciales
ynActual[:] = yn[:]
psiActual[:] = autoVectores.astype(complex)[:]


# In[13]:


print(NPasos)


# In[16]:


tiempo_inicial = time()
#run your code
for t in range(NPasos):
    if t < (tOff/deltaTiempo):
        tiempo = x_tiempo[t]
    else:
        tiempo = tOff 
    ## velocity Verlet
    aceleracionActual[:] = yn_punto_punto(ynActual[:],psiActual[:,0:espinesUp],psiActual[:,0:espinesDown],tiempo)
    ynNuevo[t,:] = ynActual [:] + velocidadActual[:] * deltaTiempo   + 1/2 * aceleracionActual[:] * deltaTiempo**2
    ynNuevo[t,:] -=   sum(ynNuevo[t,:]) / NSitios # normalizacion
    # evolucion psi usando propagador Crank-Nicholson    
    psiNuevo[:,0:espinesUp] = matriz_U((ynActual + ynNuevo[t,:])/2 ,tiempo + deltaTiempo/2) @ psiActual[:,0:espinesUp]
    aceleracionNueva[:] = yn_punto_punto(ynNuevo[t,:] , psiNuevo[:,0:espinesUp], psiNuevo[:,0:espinesDown], tiempo + deltaTiempo)[:]
    velocidadNueva[t,:] = velocidadActual[:] + 1/2 * ( aceleracionActual[:] +  aceleracionNueva[:] ) * deltaTiempo

    ## Mediciones
    #calcular rho y energia
    rho = np.zeros(NSitios)    
    for n in range(NSitios):
            rho[n] = np.vdot(psiActual[n,0:espinesUp],psiActual[n,0:espinesUp]).real
            rho[n] += np.vdot(psiActual[n,0:espinesDown],psiActual[n,0:espinesDown]).real
            rho[n] -= 1  
            # aprovechar para calcular energia
            energiaElectronesAuxiliar[n] = -1 * (1 - gamma *  ynActual[n]) * 2 * (np.exp(-1j * campoElectrico * tiempo) * ( np.vdot(psiActual[n,0:espinesUp],np.roll(psiActual,-1,axis=0)[n,0:espinesUp] ) + np.vdot(psiActual[n,0:espinesDown],np.roll(psiActual,-1,axis=0)[n,0:espinesDown] ) ) ).real
            energiaPotencialAuxiliar[n] = kaSobret0 * np.square( ynActual[n] ) / 2
            energiaCineticaAuxiliar[n] = kaSobret0 *  2 * np.square( sum(velocidadActual[0:n]) )
            energiaIonesAuxiliar[n] =  energiaPotencialAuxiliar[n] + energiaCineticaAuxiliar[n]
    # seguir con rho    
    rhoVector[t,:] = 1/ 4 * (np.roll(rho,1) + 2*rho + np.roll(rho,-1))    

    # calcular energia 
    energiaElectrones[t] = sum(energiaElectronesAuxiliar)
    energiaPotencial[t] = sum (energiaPotencialAuxiliar)
    energiaCinetica[t] = sum( energiaCineticaAuxiliar )
    energiaIones[t] = sum(energiaIonesAuxiliar)
    # Actualizar
    ynActual[:] = ynNuevo[t,:]
    velocidadActual[:] = velocidadNueva[t,:]
    psiActual[:] = psiNuevo[:]
tiempo_ejecucion = time() - tiempo_inicial
print(tiempo_ejecucion," segundos")


# In[17]:


#plt.plot(x_tiempo ,(energiaIones), label="energía iones")
plt.plot(x_tiempo ,(energiaElectrones), label="energía elec")
plt.plot(x_tiempo ,(energiaElectrones + energiaIones), label="tot_energy")
plt.xlabel("t$\omega_Q$")
plt.ylabel("energy $t_0$")
plt.legend()
#plt.savefig("energia_total_omegaQ_por_tanh.png",format="png")
plt.show


# In[31]:


print(campoElectrico)


# In[134]:


np.save("xc_variaciones_datos/yn_inicial_CN_tOff_10_tEnd_300_E_0.02_dt_0.01",yn)
np.save("xc_variaciones_datos/yn_final_CN_tOff_10_tEnd_300_E_0.02_dt_0.01",ynActual)
np.save("xc_variaciones_datos/velocidad_final_iones_CN_tOff_10_tEnd_300_E_0.02_dt_0.01",velocidadActual)
np.save("xc_variaciones_datos/aceleracion_final_iones_CN_tOff_10_tEnd_300_E_0.02_dt_0.01",aceleracionActual)
np.save("xc_variaciones_datos/energia_iones_CN_tOff_10_tEnd_300_E_0.02_dt_0.01",energiaIones)
np.save("xc_variaciones_datos/energia_electrones_CN_tOff_10_tEnd_300_E_0.02_dt_0.01",energiaElectrones)
np.save("xc_variaciones_datos/rho_vectorCN_tOff_10_tEnd_300_E_0.02_dt_0.01",rhoVector)
np.save("xc_variaciones_datos/tiempo_CN_tOff_10_tEnd_300_E_0.02_dt_0.01",x_tiempo)
np.save("xc_variaciones_datos/centro_masa_carga_CN_tOff_10_tEnd_300_E_0.02_dt_0.01",xCentroDeMasa)


# In[20]:


# Calculo de la posicion del centro de masa
theta = 2 * np.pi * np.arange(NSitios) / NSitios 
cosenoThetaPromedio = rhoVector @ np.cos( theta )    
senoThetaPromedio =  rhoVector @ np.sin( theta )
xCentroDeMasa = NSitios * np.arctan2( senoThetaPromedio , cosenoThetaPromedio ) / ( 2 * np.pi)
#xCentroDeMasa[np.argmin(xCentroDeMasa):NPasos] = xCentroDeMasa[np.argmin(xCentroDeMasa):NPasos] + NSitios
#xCentroDeMasa[np.argmax(xCentroDeMasa)+1:NPasos] = xCentroDeMasa[np.argmax(xCentroDeMasa)+1:NPasos] + NSitios
plt.plot(x_tiempo,xCentroDeMasa,".")
#plt.savefig("../notas_leandro/figs/xc_CN_tEnd_300.png")


# In[92]:


plt.plot(rhoVector[2500],label=r"$\bar{\rho}$ paso 2500")
plt.legend()
#plt.savefig("../notas_leandro/figs/cra_nic_E_0.1_rho_vector_paso_2500.png")


# In[18]:


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 100), ylim=(0, 0.07))
line, = ax.plot([], [], lw=4)
# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,
# animation function.  This is called sequentially
def animate(i):
    x = np.arange(NSitios)
    y = rhoVector[int(i * 5 / deltaTiempo),:]
    line.set_data(x, y)
    return line,
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames= 6, interval=200, blit=True)
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
#anim.save('sine_wave.gif', writer='imagemagick')
plt.show()


# In[ ]:





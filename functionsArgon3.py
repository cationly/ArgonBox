import numpy as np
from numba import jit

################################### INITIALIZATION ###################################################
def initialise(settings, seed):
    np.random.seed(seed)

    r = InitialisePositions(settings)
    v = InitialiseVelocities(settings)
    instance = md_instance(r=r[:,:,0],
                            v=v[:,:,0],
                             F=None,
                              U=None,
                               Ekin=None,
                                Virial=None,
                                 hist=None,
                                  dr=None)

    U = np.zeros(settings.NumberOfTimeSteps+1)
    Virial = np.zeros(settings.NumberOfTimeSteps+1)
    
    instance = calcForces(instance,settings)
    U[0] = instance.U
    Virial[0] = instance.Virial

    Ekin = np.zeros(settings.NumberOfTimeSteps+1)
    Ekin[0] = 0.5*np.sum(v[:,:,0] * v[:,:,0])
    instance.Ekin = Ekin[0]

    return r, v, Virial, U, Ekin, instance
    

def InitialisePositions (settings):
    NumberOfBoxesPerDimension = settings.NumberOfBoxesPerDimension
    L = settings.L
    NumberOfTimeSteps = settings.NumberOfTimeSteps
    N = settings.N

    a = L/NumberOfBoxesPerDimension
    r = np.zeros((3,N,NumberOfTimeSteps+1))
    EpsilonBoundary = 0.0001 * a #We don't want to put the particles exactly on the boundary, because it won't be clear
    #wether they belong inside the volume or the next volume. 
    GridVector = np.linspace(0,L-a,NumberOfBoxesPerDimension) #A vector used to build the fcc lattice. 
    x,y,z = np.meshgrid(GridVector,GridVector,GridVector)
    rCubenodes = (np.vstack((x.flatten(1),y.flatten(1),z.flatten(1)))).transpose()
    rShiftedInxAndy = rCubenodes + np.tile(np.array([0.5*a,0.5*a,0]),(np.size(x),1))
    rShiftedInxAndZ = rCubenodes + np.tile(np.array([0.5*a,0,0.5*a]),(np.size(x),1))
    rShiftedInyAndZ = rCubenodes + np.tile(np.array([0,0.5*a,0.5*a]),(np.size(x),1))
    rTemp = np.vstack((rCubenodes,rShiftedInxAndy,rShiftedInxAndZ,rShiftedInyAndZ)) #r is a matrix containing the positions
    #of the particles initially configured in the fcc lattice.

    rTemp += EpsilonBoundary #A small offset is added to the position of every particle to ensure that none of them are
    #located on the boundary.
    r[:,:,0] = rTemp.T
    return r

def InitialiseVelocities (settings):
    sigma = np.sqrt(settings.T)
    mu = 0
    v = np.zeros((3,settings.N,settings.NumberOfTimeSteps+1))
    VelocityGenerated = sigma * np.random.randn(3,settings.N) + mu
    AverageVelocity = np.mean(VelocityGenerated,1)
    v[:,:,0] = VelocityGenerated - AverageVelocity[:,np.newaxis]
    return v


################################### CALCULATIONS ###################################################
@jit
def calcForces(instance,settings):
    r = instance.r
    N = settings.N
    L = settings.L
    rho = settings.rho
    TruncR = settings.TruncR
 
    Fx = np.zeros((1,N))
    Fy = np.zeros((1,N))
    Fz = np.zeros((1,N))
    U = 0
    Virial = 0
    TruncR2 = TruncR * TruncR
    NumberOfBins = 200
    Delta2 = (TruncR/NumberOfBins)**2
    Delta = np.sqrt(Delta2)
    hist = np.zeros(NumberOfBins+1)

    for i in range(0,N):
        for j in range(i+1,N):

            dx = r[0,i]-r[0,j]
            dx = dx - np.rint(dx/L)*L
            
            dy = r[1,i]-r[1,j]
            dy = dy - np.rint(dy/L)*L
            
            dz = r[2,i]-r[2,j]
            dz = dz - np.rint(dz/L)*L

            R2 = dx*dx + dy*dy + dz*dz
            if (R2<TruncR2):
                
                
                
                forceMagnitude = 48*R2**(-7) -24*R2**(-4)
     
                
                Virial += -forceMagnitude*R2
                
                fx = dx * forceMagnitude
                Fx[0,i] += fx
                Fx[0,j] -= fx
                
                fy = dy * forceMagnitude
                Fy[0,i] += fy
                Fy[0,j] -= fy

                fz = dz * forceMagnitude
                Fz[0,i] += fz
                Fz[0,j] -= fz

                U += 4 * (R2**(-6) - R2**(-3))
                
                hist[int(np.sqrt(R2/Delta2))] += 1
    
    U += 8*np.pi*(N-1)*rho/3 * ( 1/(3*TruncR) - (1/TruncR**3) ) 
    
    #######################################   ↓ : due to division by zero: can be chosen arbitrary #####
    hist = 2/(rho*(N-1)) * hist/ np.hstack((10000,(4*np.pi*((np.linspace(Delta,TruncR,200))**2)*Delta) ))
    
    instance.F = np.vstack((Fx,Fy,Fz))
    instance.U = U
    instance.Virial = Virial
    instance.hist = hist

    return instance

def velocity_verlet(instance, settings):
    instance.dr = instance.v * settings.h + (instance.F/2)*settings.h**2
    instance.r = np.mod(instance.r + instance.dr,settings.L)
    instance.v = instance.v + (instance.F/2) * settings.h
    instance = calcForces(instance,settings)
    instance.v += (instance.F/2) * settings.h
    instance.Ekin = 0.5*np.sum(instance.v * instance.v)
    
    return instance



################################### DATA CONTAINERS ###################################################

class md_settings():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.N = self.NumberOfBoxesPerDimension**3 * 4
        self.L = (self.N/self.rho)**(1/3)
        self.T = self.TSI/119.8
        if(self.TruncR>self.L/np.sqrt(2)):
            print("ERROR: cutoff distance larger than 0.5*sqrt(2)-box size, trunc distance changed to (0.5+Ɛ)*sqrt(2)*L")
            self.TruncR = self.L/np.sqrt(2) * 1.0001
    def update(**kwargs):
        self.__dict__.update(kwargs)

class md_instance():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(**kwargs):
        self.__dict__.update(kwargs)
import numpy as np
import matplotlib.pyplot as plt

class AIProgress:
    def __init__(self,self_improvement_fn,optimisation_effort, plttype,top=1,precision=500,initial_Int=1,steep=None):
        self.time_fn = np.linspace(0,top,precision)
        self.I_fn = np.ones(precision)*initial_Int
        self.self_improvement_fn = self_improvement_fn
        self.optimisation_effort = optimisation_effort
        self.plttype = plttype
        self.boosts = []
        self.steep = steep
    
    def rate_of_improvement(self,I):
        """
        Yudkowsky:
            I'm not saying you literally get dy/dt = e^y that goes to infinity after finite time
            Yudkowsky has us abruptly switch from y' = y to y' = e^y once we hit the threshold where RSI is possible.
            For continuous takeoff we model always the same equation y' = Ay + f(y)*e^y
        """
        #return self.optimisation_effort*I + (self.self_improvement_fn(I)*np.exp(I)), self.self_improvement_fn(I)

        """
        Bostrom:
            dI/dt = cI vs dI/dt = (c+I)I = cI + I^2 'when the AI reaches the crossover point'
            where difference is modelled as cI + f(I)*I^2
        """
        derivative = self.optimisation_effort*I + (self.self_improvement_fn(I,self.steep)*I**2)
        derivative_0 = self.optimisation_effort*I + (self.self_improvement_fn(I)*I**2)
        if self.steep is not None:
            return derivative, self.self_improvement_fn(I,self.steep)*I
        return derivative_0, self.self_improvement_fn(I,self.steep)*I


        "no assumption of exponential progress by default, 'constant underlying increase'"
        #return self.optimisation_effort + (self.self_improvement_fn(I)*I), self.self_improvement_fn(I)
    
    def simulate(self):
        dt = self.time_fn[1]
        for i,t in enumerate(self.time_fn):
            I_approx = self.I_fn[i-1]
            dIdT, boost = self.rate_of_improvement(I_approx)
            #print(t,I_approx,dIdT)
            self.I_fn[i] = I_approx + dIdT*dt
            self.boosts.append(boost)
            #print(t,boost)
    
    def doubling_intervals(self):
        Intelligence = self.I_fn[0]
        doubling_intervals = [[Intelligence,0]]
        for i,I_t in enumerate(self.I_fn):
            time = i * self.time_fn[1]
            if I_t > Intelligence*2:
                interval = time
                Intelligence = I_t
                doubling_intervals.append([Intelligence,interval])
        return doubling_intervals

    def get_results(self,ax1,ax2):
        self.simulate()
        ax1.plot(self.time_fn,self.I_fn, label=self.plttype)
        ax2.plot(self.time_fn,self.boosts,label=self.plttype)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def yud_function(Intelligence):
    return (Intelligence > AGI)*scale

def continuous_poly_function(Intelligence):
    if Intelligence<AGI:
        return ((Intelligence/AGI)**N)*scale
    else:
        return scale

def continuous_function(Intelligence,steep=None):
    global k
    if steep is not None:
        sharpness = steep
    else:
        sharpness = k

    if sharpness > 0:
        return sigmoid(sharpness*(Intelligence-AGI))*scale
    else:
        return 0

def diminishing_returns_function(Intelligence,steep=None):
    global k
    if steep is not None:
        sharpness = steep
    else:
        sharpness = k

    if sharpness > 0:
        return (sigmoid(sharpness*(Intelligence-AGI))*scale)/Intelligence
    else:
        return 0

def slow_function(Intelligence):
    return 0

def superintelligence_box_function(Intelligence):
    return scale


if __name__ == "__main__":
    I_AGI = AGI = 4 #Intelligence level at which AI reaches its maximum ability to exert improvements on itself
    s = scale = 2 #how strong (in absolute size) is an AGI's RSI ability?
    c = effort = 1 #initial optimisation effort applied to AGI
    #N = 2 #how sudden is the continuous change - higher is higher order polynomial
    I_0 = IniInt = 0.5 #Initial capability of the AI system
    krange = np.hstack((np.array([0.0,1.0,2.0]),np.array(1e10)))
    k=1
    d=k #steepness of logistic curve

    top = 2.5
    precision = 2000
    YMAX = 5*AGI
    k=1 #steepness of logistic curve
    #krange = (np.arange(4)*2)[1:]

    infostring = "s={}, c={}, I_AGI = {}, I_0 = {}_".format(scale,effort,AGI,IniInt)

    t = np.linspace(0,2*AGI,precision)
    plt.plot(t,[yud_function(t1) for t1 in t],label='yudkowsky')
    for k in krange:
        plt.plot(t,[continuous_function(t1) for t1 in t],label='Continuous, '+str(k))
    for k in krange:
        plt.plot(t,[diminishing_returns_function(t1) for t1 in t],label='Dim_R, '+str(k))
    
    plt.ylim(0,2)
    #for k in krange:
    #    plt.plot(t,[diminishing_returns_function(t1) for t1 in t],label='Dim-R, '+str(k))

    plt.plot(t,[slow_function(t1) for t1 in t],label='slow')
    plt.xlabel('AI Capability I')
    plt.ylabel('RSI function f(I)')
    plt.legend()
    plt.savefig(infostring+"RSI-able.png")

    fig,(axis1,axis2) = plt.subplots(2,sharex=True,figsize=(10,10))
    plt.xlabel("Time")
    axis1.set_ylabel("AI Capability I")
    axis2.set_ylabel("")

    axis1.plot(np.linspace(0,top,precision),AGI+np.zeros_like(np.linspace(0,top,precision)),'--',label = 'AGI')
    axis1.set_title(infostring[:-1])

    #print('Discontinuous')
    #progress = AIProgress(yud_function,effort,'Discontinuous',top,precision,IniInt)
    #progress.get_results(axis1,axis2)

    print('Continuous')
    for k in krange:
        progress2 = AIProgress(continuous_function,effort, 'const returns, d=' +str(k),top,precision,IniInt)
        progress3 = AIProgress(diminishing_returns_function,effort, 'dim returns, d=' +str(k),top,precision,IniInt)
        progress2.get_results(axis1,axis2)
        progress3.get_results(axis1,axis2)

    
    #print('Slow')
    #progress3 = AIProgress(slow_function,effort,'Slow',top,precision,IniInt)
    #progress3.get_results(axis1,axis2)

    axis1.axis(ymax=YMAX,ymin=0)
    axis2.axis(ymax=YMAX,ymin=0)
    #plt.yscale("linear")
    #plt.ylim(1,1e3)
    axis1.legend()
    axis2.legend()
    plt.savefig(infostring+"Progress_lin.png")
    axis1.set_yscale('log')
    axis2.set_yscale('log')
    axis1.axis(ymax=YMAX**2)
    axis2.axis(ymax=YMAX**2)
    axis2.set_ylabel("f(I)I")
    plt.savefig(infostring+"Progress_log.png")


    """
    print('Superintelligence_Fig')
    fig,(axis1,axis2) = plt.subplots(2,sharex=True,figsize=(16,9))
    plt.xlabel("Time")
    axis1.set_ylabel("AI Capability")
    axis2.set_ylabel("RSI-able Fraction")

    axis1.set_title("Superintelligence Box Figure".format(scale,effort,AGI,IniInt))

    progress4 = AIProgress(superintelligence_box_function,effort,'Superintelligence Box Figure',top,precision,IniInt)
    progress4.get_results(axis1,axis2)
    axis1.set_yscale('log')
    axis1.axis(ymax=YMAX*1e8,xmax=1,xmin=0)
    plt.savefig("Superintelligence p77")
    plt.show()
    """
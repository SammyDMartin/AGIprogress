import numpy as np
import matplotlib.pyplot as plt

"Intended to address the 'narrow window argument' for takeoff speeds: https://wiki.issarice.com/wiki/Narrow_window_argument_against_continuous_takeoff"

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
        if self.steep is not None:
            return self.optimisation_effort*I + (self.self_improvement_fn(I,self.steep)*I**2), self.self_improvement_fn(I,self.steep)
        return self.optimisation_effort*I + (self.self_improvement_fn(I)*I**2), self.self_improvement_fn(I)


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
        ax2.plot(self.time_fn,self.boosts,label=self.plttype+("-RSI fraction"))






"""
VARIABLES

"""

scale = 1 #how strong (in absolute size) is an AGI's RSI ability?
effort = 1 #outisde optimisation
AGI = 4 #Intelligence level at which AI reaches its maximum ability to exert improvements on itself
IniInt = 1 #Initial capability of the AI system
krange = np.hstack((np.arange(5)/2,np.array(1e10))) #the range of different continuous progress scenarios








def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def continuous_function(Intelligence):
    if d == 0:
        return 0
    return sigmoid(d*(Intelligence-AGI))*scale

if __name__ == "__main__":
    plt.style.use('fivethirtyeight')
    #plt.rcParams.update({'font.size': 10})
    top = 2.5
    precision = 2000
    YMAX = 5*AGI
    #krange = (np.arange(4)*2)[1:]

    t = np.linspace(0,2*AGI,precision)
    #plt.plot(t,[yud_function(t1) for t1 in t],label='yudkowsky')
    fig,(axis1) = plt.subplots(1,sharex=True,figsize=(16,9))
    for d in krange:
        plt.plot(t,[continuous_function(t1) for t1 in t],label='d='+str(d))
    #plt.plot(t,[slow_function(t1) for t1 in t],label='slow')
    axis1.set_xlabel('I')
    axis1.set_ylabel('f(I)')
    axis1.legend()
    fig.savefig("RSI-able.png")

    #fig,(axis1,axis2) = plt.subplots(2,sharex=True,figsize=(16,9))
    fig,(axis1) = plt.subplots(1,sharex=True,figsize=(16,9))
    fig2,(axis2) = plt.subplots(1,sharex=True,figsize=(16,9))
    axis1.set_xlabel("Time")
    axis1.set_ylabel("I(t)")
    axis2.set_ylabel("RSI-able Fraction")

    axis1.plot(np.linspace(0,top,precision),AGI+np.zeros_like(np.linspace(0,top,precision)),'--',label = 'AGI')
    axis1.set_title("s={}, c={}, I_AGI = {}, I_0 = {}".format(scale,effort,AGI,IniInt))
    #print('Discontinuous')
    #progress = AIProgress(yud_function,effort,'Discontinuous',top,precision,IniInt)
    #progress.get_results(axis1,axis2)
    print('Continuous')
    for d in krange:
        progress2 = AIProgress(continuous_function,effort, 'd=' +str(d),top,precision,IniInt)
        progress2.get_results(axis1,axis2)

    #print('Slow')
    #progress3 = AIProgress(slow_function,effort,'Slow',top,precision,IniInt)
    #progress3.get_results(axis1,axis2)

    axis1.axis(ymax=YMAX,ymin=0)
    #plt.yscale("linear")
    #plt.ylim(1,1e3)
    axis1.legend()
    axis2.legend()
    #fig.savefig(str(effort)+str(AGI)+str(IniInt)+"_Progress_lin.png")
    axis1.set_yscale('log')
    axis1.axis(ymax=YMAX**2)
    fig.savefig(str(scale)+str(effort)+str(AGI)+str(IniInt)+"_Progress_log.png")
    #fig2.savefig(str(effort)+str(AGI)+str(IniInt)+"RSI-ABLE.png")
    plt.show()
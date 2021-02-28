from progress import AIProgress
from progress import sigmoid
from progress import continuous_function
import numpy as np
import matplotlib.pyplot as plt
from progress import krange, IniInt, effort, scale, AGI #last two are just for show - are set in continuous in other file

"""There will be a complete 4 year interval in which world output doubles,
before the first 1 year interval in which world output doubles. (Similarly, we’ll see an 8 year doubling before a 2 year doubling, etc.)"""

steeps = krange
initial_intelligence = IniInt
opt=effort
#ensures results come out the same as other file

plt.figure(figsize=(20,20))

#produces plots of doubling time over time for different speeds of continuous progress
for cont_steep in steeps:
    progress = AIProgress(continuous_function,optimisation_effort=opt,plttype=str(cont_steep),top=20,precision=10000,initial_Int=initial_intelligence,steep=cont_steep)
    progress.simulate()
    intervals = progress.doubling_intervals()
    print("Rate of Gain = "+str(cont_steep))
    print(str("Intelligence").rjust(30),str("Multiple (should be 2)").rjust(30),str("dT"))
    differences = []
    for i,tup in enumerate(intervals):
        multiple = tup[0]/intervals[i-1][0]
        if i > 0:
            difference = tup[1]-intervals[i-1][1]
            differences.append(difference)
            init_d = differences[0]
            print(str(round(tup[0],1)).rjust(30),str(round(multiple,1)).rjust(30),str(round(difference/init_d,3)))

    differences = [d/init_d for d in differences] #comment out for absolute doubling times
    label = str("Discontinuity Level: ")+str(cont_steep)+str(", First Doubling time: ")+str(round(init_d,2))
    plt.bar(np.arange(len(differences)),differences,label=label)
    plt.xlabel("Doubling Number")
    plt.ylabel("Time/First Doubling")
    plt.title("inside/outside opt={},{}, AGI = {}, start opt = {}".format(scale,opt,AGI,initial_intelligence))

plt.xlim(0,10)
plt.legend()
plt.savefig("Doubling.png")
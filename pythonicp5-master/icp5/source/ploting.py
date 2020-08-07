from scipy import stats #import scipy
import numpy as np  #import numpy
from matplotlib import pyplot as plt
x=np.array([0,1,2,3,4,5,6,7,8,9]) #write the values of x in a list
y=np.array([1,3,2,5,7,8,8,9,10,12]) # write the values
slope ,intercept,r_value,p_value,std_err=stats.linregress(x,y) # we get slope and intercept

plt.plot(x,y,'ro',color='black') # the coordinates will be given black color
plt.ylabel('y')   #print ylabel
plt.xlabel('x')   #print xlabel
plt.axis([0,10,0,13])  # here we need to give the range of x axis and y-axis
plt.plot(x,x*slope+intercept,'r') #
plt.plot() 
plt.show() # it will show the graph
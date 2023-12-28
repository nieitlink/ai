import numpy as np
def nonlin(x, deriv=False):
  if(deriv==True):
    return x* (1-x)
  return 1/(1+np.exp(-x))

xrange=10000
X = np.array([ [0,0,1], [0,1,1], [1,0,1], [1,1,1]])
Y = np.array([[0,0,1,1]]).T
np.random.seed (1)
syn0 = 2*np.random.random((3,1)) -1
for iter in range (xrange):
  a = X
  b = nonlin (np.dot (a, syn0))
  c = Y-b
  d = c * nonlin (b, True)
  syn0 += np.dot(a.T,d)
print("Actual Output:\n"+str(Y))
print ("Output After Training:")
print (b)

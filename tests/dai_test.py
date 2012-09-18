import sys
sys.path.append('./build')
sys.path.append('../lib')
from probability import *

u = PyVar(0,2)
v = PyVar(1,2)

x = PyVarSet(u,v)
y = PyVarSet(u)

print len(x)
print len(y)

f1 = PyFactor(x)
f2 = PyFactor(u)

f1.set(0,0.01)

print f1
f1.normalize()
print f1

fg = PyFactorGraph(f1)

jt = PyJTree(fg)

print jt

#########################################
vars = []
facs = []
for i in range(50):
	vars.append(PyVar(i,2))
	facs.append(PyFactor(vars[i]))

#vs1 = PyVarSet(*vars[:3])
#fac1 = PyFactor(vs)
print facs[0].get(1)
fg = PyFactorGraph(*facs)
jt = PyJTree(fg)

jt.init()
jt.run()
print jt.iterations()

print jt.calcMarginal(PyVarSet(vars[0]))



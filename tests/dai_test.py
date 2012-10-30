import sys
sys.path.append('./build')
sys.path.append('../lib')
from samcnet.probability import *
from samcnet.pydai import PyVar, PyVarSet, PyFactor, PyJTree

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

jt = PyJTree(f1)

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
jt = PyJTree(*facs)

jt.init()
jt.run()
print jt.iterations()

print jt.marginal(PyVarSet(vars[0], vars[1]))

node1 = CPD(0, 2, {0:0.25, 1:0.9}, {1:2})
node2 = CPD(1, 2, {(): 0.25}) 
j = node1*node2

g = GroundNet(j)

g.kld(j)

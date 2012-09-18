from probability import GroundNet, JointDistribution, CPD
from bayesnetcpd import BayesNetCPD
from pydai import PyVar, PyVarSet, PyFactor, PyJTree

import numpy as np

C = CPD(0, 2, {(): np.r_[0.5]})
S = CPD(1, 2, {(0,): np.r_[0.5], (1,): np.r_[0.9]}, {0:2})
R = CPD(2, 2, {(0,): np.r_[0.8], (1,): np.r_[0.2]}, {0:2})
W = CPD(3, 2, {(0,0): np.r_[0.9], (0,1): np.r_[0.1], (1,0): np.r_[0.1], (1,1): np.r_[0.01]}, {1:2, 2:2})

C2 = CPD(0, 2, {(): np.r_[0.999]})
W2 = CPD(3, 2, {(0,0): np.r_[0.5], (0,1): np.r_[0.5], (1,0): np.r_[0.5], (1,1): np.r_[0.51]}, {1:2, 2:2})
#W = CPD(3, 2, {(0,0): np.r_[1.0], (0,1): np.r_[0.1], (1,0): np.r_[0.1], (1,1): np.r_[0.01]}, {1:2, 2:2})

j = JointDistribution(cpds=[C,S,R,W])
j2 = JointDistribution(cpds=[C2,S,R,W])
#j3 = JointDistribution(cpds=[C,S,R,W2])

###########################################################
#v = PyVar(0,2)
#x = PyFactor(v)
#j = PyJTree(x)

#j.init()
#j.run()
#print j.marginal_array(0)

#x = PyFactor(PyVar(0,3))
#j = PyJTree(x)
#print j.marginal_array(0)

#print j.marginal_array(0)

###########################################################

gn = GroundNet(j)
gn2 = GroundNet(j2)
#gn3 = GroundNet(j3)
#import gc
##for i in range(5):
#print gc.get_count()
#y = PyVarSet(PyVar(0,2))
    ##print id(x)
#gc.collect()
#print len(y)
    #print x
    #print id(y)
    #print y

    #print id(gn.makeFactor(W))

#for i in range(5):
    #print gn.makejtree(j2).marginal_array(0)
    #print gn.makejtree(j).marginal_array(0)

#print gn.belief(3)
print gn.marginal_array(0,3)
print gn2.marginal_array(0,3)
#print gn.kld(j2)
#print gn.kld(j)

#for i in range(100000):
    #gn.kld(j2)

#for i in range(100000):
    #gn.makejtree(j2)

#import gc
#print gc.get_count()
#x = gn.makeFactor(W)
#print gc.get_count()
#gc.collect()
#print gc.get_count()

#import gc
#print gc.get_count()
#x = PyVar(0,2)
#print type(x)
#print gc.get_count()
#del x
#del gn
#print gc.get_count()
#gc.collect()
#print gc.get_count()

#for i in xrange(500000):
    #x = gn.makeFactor(W)
    #x.clean()

#for i in range(1000000):
    #gn.belief_array(0,1)

#print 'iterations: %d' % gn.jtree.iterations()
#print gn.belief(0)
#print gn.belief(1)
#print gn.belief(2)
#print gn.belief(3)

#C = dai.Var(0, 2)  # Define binary variable Cloudy (with label 0)
#S = dai.Var(1, 2)  # Define binary variable Sprinkler (with label 1)
#R = dai.Var(2, 2)  # Define binary variable Rain (with label 2)
#W = dai.Var(3, 2)  # Define binary variable Wetgrass (with label 3)

## Define probability distribution for C
#P_C = dai.Factor(C)
#P_C[0] = 0.5            # C = 0
#P_C[1] = 0.5            # C = 1

## Define conditional probability of S given C
#P_S_given_C = dai.Factor(dai.VarSet(S,C))
#P_S_given_C[0] = 0.5    # C = 0, S = 0
#P_S_given_C[1] = 0.9    # C = 1, S = 0
#P_S_given_C[2] = 0.5    # C = 0, S = 1
#P_S_given_C[3] = 0.1    # C = 1, S = 1

## Define conditional probability of R given C
#P_R_given_C = dai.Factor(dai.VarSet(R,C))
#P_R_given_C[0] = 0.8    # C = 0, R = 0
#P_R_given_C[1] = 0.2    # C = 1, R = 0
#P_R_given_C[2] = 0.2    # C = 0, R = 1
#P_R_given_C[3] = 0.8    # C = 1, R = 1

## Define conditional probability of W given S and R
#SRW = dai.VarSet(S,R)
#SRW.append(W)
#P_W_given_S_R = dai.Factor(SRW)
#P_W_given_S_R[0] = 1.0  # S = 0, R = 0, W = 0
#P_W_given_S_R[1] = 0.1  # S = 1, R = 0, W = 0
#P_W_given_S_R[2] = 0.1  # S = 0, R = 1, W = 0
#P_W_given_S_R[3] = 0.01 # S = 1, R = 1, W = 0
#P_W_given_S_R[4] = 0.0  # S = 0, R = 0, W = 1
#P_W_given_S_R[5] = 0.9  # S = 1, R = 0, W = 1
#P_W_given_S_R[6] = 0.9  # S = 0, R = 1, W = 1
#P_W_given_S_R[7] = 0.99 # S = 1, R = 1, W = 1

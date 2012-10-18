# cython: profile=False
cimport cython
from libc.math cimport exp, ceil, floor

import sys
import os
from itertools import izip

import numpy as np
cimport numpy as np

cdef class SAMCRun:
    cdef public:
        object obj, db, refden, hist, indicator, mapvalue, ground
        int lowEnergy, highEnergy, grid, accept_loc, total_loc, iteration, burn, stepscale
        double rho, tau, mapenergy, delta, scale
    def __init__(self, obj, ground=None, burn=100000, stepscale = 10000):

        self.obj = obj # Going to be a BayesNet for now, but we'll keep it general
        self.clear()

        self.scale = 1
        self.set_energy_limits()

        self.rho=1.0
        self.tau=1.0;

        self.burn = burn
        self.stepscale = stepscale

        self.ground = ground

    def set_energy_limits(self):
        cdef int i
        cdef double oldenergy, energy, low, high, spread

        if self.iteration > 0:
            self.lowEnergy = <int> self.hist[0,0]
            self.highEnergy = <int> self.hist[0,-1]
            return

        print "Establishing energy limits... ",
        low = high = oldenergy = energy = self.obj.energy()
        while high > 1e90:
            self.obj.propose()
            low = high = oldenergy = energy = self.obj.energy()
        for i in range(2000):
            self.obj.propose()
            energy = self.obj.energy() 
            r = oldenergy-energy / 20.0 # Higher temperature for exploration
            if r > 0.0 or np.random.rand() < exp(r):
                if energy < low:
                    low = energy
                elif energy > high:
                    high = energy
                oldenergy = energy
            else:
                self.obj.reject()

        spread = high - low
        #if spread < 1000:
            #spread = 1000
        low = floor(low - .6 * spread)
        high = ceil(high)
        if low < 0:
            low = 0

        print "Done. Setting limits to (%d, %d)" % (low,high)

        spread = high - low
        self.scale = spread/500
        print "Setting scale to %f" % (self.scale)

        self.lowEnergy = <int>low
        self.highEnergy = <int>high
        self.grid = <int>ceil((self.highEnergy - self.lowEnergy) / self.scale)

        #self.refden = np.arange(self.grid, 0, -1, dtype=np.double)
        #self.refden = self.refden**2
        self.refden = np.ones(self.grid, dtype=np.double)

        self.refden /= self.refden.sum()
        self.hist = np.zeros((3,self.grid), dtype=np.double)

        self.hist[0,:] = np.arange(self.lowEnergy, self.highEnergy, self.scale)
        self.indicator = np.zeros((self.grid),dtype=np.int32) # Indicator is whether we have visited a region yet

    def clear(self):
        self.db = None
        self.mapenergy = np.inf
        self.mapvalue = None
        self.delta = 1.0
        self.iteration = 0
        self.accept_loc = 0
        self.total_loc = 0

    def estimate_func_mean(self, trunc=None):
        """ 
        Using the function of interest in the object, estimate the mean of the function
        on the random weighted samples.
        """
        assert self.db != None
        thetas = self.db['thetas']
        assert thetas.shape[0] != 0

        if trunc:
            num = len(self.db)/trunc
            grab = thetas.argsort()[::-1][num:]
            thetas = thetas[grab]

        part = np.exp(thetas - thetas.max())

        if trunc:
            numerator = (part * self.db['funcs'][grab]).sum()
        else:
            numerator = (part * self.db['funcs']).sum()
        denom = part.sum()
        print "Calculating function mean: %g / %g = %g." % (numerator, denom, numerator/denom)
        return numerator / denom

    cdef find_region(self, energy):
        cdef int i
        if energy > self.highEnergy: 
            return self.grid-1
        elif energy < self.lowEnergy:
            return 0
        else: 
            i = <int> floor((energy-self.lowEnergy)/self.scale)
            return i if i<self.grid else self.grid-1

    #@cython.boundscheck(False) # turn off bounds-checking for entire function
    def sample(self, int iters, double temperature = 1.0):
        cdef int current_iter, accept, oldregion, newregion, i, nonempty, dbsize, scheme
        cdef double oldenergy, newenergy, r, un
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indicator = \
                self.indicator
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] locfreq = \
                np.zeros((self.grid,), dtype=np.int32)
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] hist = \
                self.hist[1].copy()
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] refden = \
                self.refden
        oldenergy = self.obj.energy()
        oldregion = self.find_region(oldenergy) # AKA nonempty
        indicator[oldregion] = 1

        scheme_detail = {}
        scheme_detail[1] = [0]*2
        scheme_detail[2] = [0]*2
        scheme_detail[3] = [0]*2

        dbsize = self.iteration + int(iters) - self.burn
        if dbsize < 0:
            dbsize = 0
        self.db = self.obj.init_db(self.db, dbsize)
        print("Initial Energy: %g" % oldenergy)
        #fid = open("rlogpy2",'w')

        for current_iter in range(self.iteration, self.iteration + int(iters)):
            self.iteration += 1

            #self.delta = temperature * float(self.stepscale) / max(self.stepscale, self.iteration)
            self.delta = float(self.stepscale) / max(self.stepscale, self.iteration)

            scheme = self.obj.propose()
            newenergy = self.obj.energy()

            if newenergy < self.mapenergy: # NB: Even if not accepted
                self.mapenergy = newenergy
                self.mapvalue = self.obj.copy()
    
            ####### acceptance of new moves #########

            newregion = self.find_region(newenergy)

            indicator[newregion] = 1

            r = hist[oldregion] - hist[newregion] + (oldenergy-newenergy) /temperature
            #print hist[oldregion] - hist[newregion], (oldenergy-newenergy)
            
            if r > 0.0 or np.random.rand() < exp(r):
                accept=1
            else:
                accept=0;

            #fid.write("%f,%f,%f,%f,%f,%f\n" % (hist[oldregion], hist[newregion], oldenergy,
                #newenergy, r, self.obj.lastscheme))
            
            #print("%d\t r:%f\t oldregion:%d\t newregion:%d\thist[old]:%f\t hist[new]:%f fold:%f, fnew:%f" %
                    #(accept, r,oldregion, newregion, hist[oldregion], 
                        #hist[newregion], oldenergy, newenergy))


            if accept == 0:
                self.hist[2,oldregion] += 1.0
                self.obj.reject()
                self.total_loc += 1
                scheme_detail[scheme][0] += 1
            elif accept == 1:
                self.hist[2,newregion] += 1.0
                self.accept_loc += 1
                self.total_loc += 1
                oldregion = newregion
                oldenergy = newenergy
                scheme_detail[scheme][0] += 1
                scheme_detail[scheme][1] += 1
                  
            locfreq[oldregion] += 1
            hist += self.delta*(locfreq-refden)
            locfreq[oldregion] -= 1


            if current_iter >= self.burn:
                self.obj.save_to_db(self.db, 
                        hist[oldregion], 
                        oldenergy, 
                        current_iter-self.burn, 
                        self.ground)

            if self.iteration % 10000 == 0:
                print("Iteration: %8d, delta: %5.2f, best energy: %7g, current energy: %7g" % \
                        (self.iteration, self.delta, self.mapenergy, newenergy))

        self.hist[1] = hist
        self.indicator = indicator

        ###### Calculate summary statistics #######
        print("Accept_loc: %d" % self.accept_loc)
        print("Total_loc: %d" % self.total_loc)
        print("Acceptance: %f" % (float(self.accept_loc)/float(self.total_loc)))

        return scheme_detail


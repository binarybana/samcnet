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
        object obj, db, refden, hist, mapvalue
        int lowEnergy, highEnergy, grid, accept_loc, total_loc, iteration, burn, stepscale, thin
        double rho, tau, mapenergy, delta, scale, refden_power
    def __init__(self, obj, burn=100000, stepscale = 10000, refden=0.0, thin=1):

        self.obj = obj # Going to be a BayesNet for now, but we'll keep it general
        self.clear()

        self.scale = 1
        self.rho=1.0
        self.tau=1.0;

        self.burn = burn
        self.stepscale = stepscale
        self.refden_power = refden
        self.thin = thin

        self.db = None

        self.set_energy_limits()

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
            r = (oldenergy-energy) / 3.0 # Higher temperature for exploration
            if r > 0.0 or np.random.rand() < exp(r):
                if energy < low:
                    low = energy
                elif energy > high:
                    high = energy
                oldenergy = energy
            else:
                self.obj.reject()

        spread = high - low
        low = floor(low - (0.6 * spread))
        high = ceil(high + (0.2 * spread))

        print "Done. Setting limits to (%d, %d)" % (low,high)

        spread = high - low
        self.scale = max(0.25, spread/500.0)
        print "Setting scale to %f" % (self.scale)

        self.lowEnergy = <int>low
        self.highEnergy = <int>high

        self.grid = <int>ceil((self.highEnergy - self.lowEnergy) / self.scale)

        self.refden = np.arange(self.grid, 0, -1, dtype=np.double)**self.refden_power
        self.refden /= self.refden.sum()

        self.hist = np.zeros((3,self.grid), dtype=np.double)
        self.hist[0,:] = np.arange(self.lowEnergy, self.highEnergy, self.scale)

    def clear(self):
        self.db = None
        self.mapenergy = np.inf
        self.mapvalue = None
        self.delta = 1.0
        self.iteration = 0
        self.accept_loc = 0
        self.total_loc = 0

    def save_to_db(self, double theta, double energy, int iteration):
        assert self.db is not None, 'DB None when trying to save sample.'
        self.db[iteration][0] = theta
        self.db[iteration][1] = energy
        self.db[iteration][2] = self.obj.save_to_db()

    def func_cummean(self, accessor=None):
        """ 
        Using the function of interest in the object, compute the cumulative mean of the function
        on the random weighted samples.

        If accessor is given, then it is a function which pulls out the quantity of interest from
        db['funcs'].
        """
        assert self.db != None, 'db not initialized'
        assert len(self.db) != 0, 'Length of db is zero! Perhaps you have not "\
                "proceeded beyond the burn-in period'

        thetas = self.db['thetas']
        if accessor == None:
            funcs = self.db['funcs'].astype(np.float)
        else:
            funcs = np.vectorize(accessor)(self.db['funcs']).astype(np.float)

        part = np.exp(thetas - thetas.max())
        numerator = (part * funcs).cumsum()
        denom = part.cumsum()
        return numerator / denom

    def func_mean(self, accessor=None, trunc=None):
        """ 
        Using the function of interest in the object, estimate the mean of the function
        on the random weighted samples.

        If trunc is given, then it gives the proportion of samples (in descending order by theta) 
        to discard.

        If accessor is given, then it is a function which pulls out the quantity of interest from
        db['funcs'].
        """
        assert self.db != None, 'db not initialized'
        assert len(self.db) != 0, 'Length of db is zero! Perhaps you have not "\
                "proceeded beyond the burn-in period'
        thetas = self.db['thetas']
        if accessor == None:
            funcs = self.db['funcs'].astype(np.float)
        else:
            funcs = np.vectorize(accessor)(self.db['funcs']).astype(np.float)

        if trunc:
            num = len(self.db)/trunc
            grab = thetas.argsort()[::-1][num:]
            thetas = thetas[grab]

        part = np.exp(thetas - thetas.max())

        if trunc:
            numerator = (part * funcs[grab]).sum()
        else:
            numerator = (part * funcs).sum()
        denom = part.sum()
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

    def init_db(self, size):
        dtype = [('thetas',np.double),
                 ('energies',np.double),
                 ('funcs',np.object)]
        if self.db == None:
            self.db = np.zeros(size, dtype=dtype)
        elif self.db.shape[0] != size:
            self.db = np.resize(self.db, size)
        else:
            raise Exception("DB not initialized!")


    #@cython.boundscheck(False) # turn off bounds-checking for entire function
    def sample(self, int iters, double temperature = 1.0):
        cdef int current_iter, accept, oldregion, newregion, i, nonempty, dbsize
        cdef double oldenergy, newenergy, r, un
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] locfreq = \
                np.zeros((self.grid,), dtype=np.int32)
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] hist = \
                self.hist[1].copy()
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] refden = \
                self.refden
        oldenergy = self.obj.energy()
        oldregion = self.find_region(oldenergy) # AKA nonempty

        dbsize = (self.iteration + int(iters) - self.burn)//self.thin
        if dbsize < 0:
            dbsize = 0
        self.init_db(dbsize)
        
        print("Initial Energy: %g" % oldenergy)

        for current_iter in range(self.iteration, self.iteration + int(iters)):
            self.iteration += 1

            self.delta = temperature * float(self.stepscale) / max(self.stepscale, self.iteration)

            self.obj.propose()
            newenergy = self.obj.energy()

            if newenergy < self.mapenergy: # NB: Even if not accepted
                self.mapenergy = newenergy
                self.mapvalue = self.obj.copy()
    
            ####### acceptance of new moves #########

            newregion = self.find_region(newenergy)

            r = hist[oldregion] - hist[newregion] + (oldenergy-newenergy) #/temperature
            
            if r > 0.0 or np.random.rand() < exp(r):
                accept=1
            else:
                accept=0;

            if accept == 0:
                self.hist[2,oldregion] += 1.0
                self.obj.reject()
                self.total_loc += 1
            elif accept == 1:
                self.hist[2,newregion] += 1.0
                self.accept_loc += 1
                self.total_loc += 1
                oldregion = newregion
                oldenergy = newenergy
                  
            locfreq[oldregion] += 1
            hist += self.delta*(locfreq-refden)
            locfreq[oldregion] -= 1

            if current_iter >= self.burn and current_iter % self.thin == 0:
                self.save_to_db(hist[oldregion], 
                                oldenergy, 
                                (current_iter-self.burn)//self.thin)

            if self.iteration % 10000 == 0:
                print("Iteration: %8d, delta: %5.2f, best energy: %7g, current energy: %7g" % \
                        (self.iteration, self.delta, self.mapenergy, newenergy))

        self.hist[1] = hist

        ###### Calculate summary statistics #######
        print("Accept_loc: %d" % self.accept_loc)
        print("Total_loc: %d" % self.total_loc)
        print("Acceptance: %f" % (float(self.accept_loc)/float(self.total_loc)))


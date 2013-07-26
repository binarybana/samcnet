# cython: profile=True
cimport cython
from libc.math cimport exp, ceil, floor

import sys
import os
import tempfile
import zlib
import tables as t
from itertools import izip

import numpy as np
cimport numpy as np

cdef class SAMCRun:
    cdef public:
        object obj, db, refden, hist, mapvalue, verbose
        int lowEnergy, highEnergy, grid, accept_loc, total_loc, iteration, burn, stepscale, thin
        double mapenergy, delta, scale, refden_power
    def __init__(self, obj, burn=100000, stepscale = 10000, refden=0.0, thin=1, verbose=False,
            lim_iters=2000):
        self.verbose = verbose
        self.obj = obj # Going to be a BayesNet for now, but we'll keep it general
        self.clear()

        self.scale = 1
        self.burn = burn
        self.stepscale = stepscale
        self.refden_power = refden
        self.thin = thin

        self.db = None

        self.set_energy_limits(lim_iters)

    def set_energy_limits(self, int lim_iters):
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
        for i in range(lim_iters):
            self.obj.propose()
            energy = self.obj.energy() 
            r = (oldenergy-energy) / 3.0 # Higher temperature for exploration
            if r > 0.0 or np.random.rand() < exp(r):
                if self.verbose:
                    print("Energy of {} accepted".format(energy))
                if energy < low:
                    low = energy
                elif energy > high:
                    high = energy
                oldenergy = energy
            else:
                self.obj.reject()

        spread = high - low
        low = int(floor(low - (0.6 * spread)))
        high = int(ceil(high + (0.2 * spread)))

        print "Done. Setting limits to (%d, %d)" % (low,high)

        spread = high - low
        self.scale = max(0.25, spread/100.0)
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

    def init_db(self, size):
        if self.db == None:
            filt = t.Filters(complib='bzip2', complevel=7, fletcher32=True)
            if not os.path.exists('.tmp'):
                print("Creating temp directory: .tmp")
                os.mkdir('.tmp')
            name = tempfile.mktemp(prefix='samc', dir='.tmp')
            self.db = t.openFile(name, mode = 'w', title='SAMC Run Data', filters=filt)

            self.db.createGroup('/', 'samc', 'SAMC info', filters=filt)
            self.db.createEArray('/samc', 'theta_trace', t.Float64Atom(), (0,), expectedrows=size)
            self.db.createEArray('/samc', 'energy_trace', t.Float64Atom(), (0,), expectedrows=size)
            self.db.createCArray('/samc', 'theta_hist', t.Float64Atom(), (self.grid,))
            self.db.createCArray('/samc', 'freq_hist', t.Int64Atom(), (self.grid,)) 

            objdb = self.db.createGroup('/', 'object', 'Object info', filters=filt)
            samples = self.db.createGroup(objdb, 'samples', 'Samples')
            objfxn = self.db.createGroup(objdb, 'objfxn', 'Objective function outputs')
            self.db.root.samc._v_attrs.temperature = []

            self.obj.init_db(self.db, size)

    def read_db(self):
        assert self.db.isopen == 1, "DB not open!"
        fname = self.db.filename
        self.db.close()
        fid = open(fname, 'r')
        data = zlib.compress(fid.read())
        fid.close()
        self.db = t.openFile(fname, 'r+')
        return data

    def close_db(self):
        self.db.close()

    def save_iter_db(self, double theta, double energy, int iteration):
        self.db.root.samc.theta_trace.append((theta,))
        self.db.root.samc.energy_trace.append((energy,))

        self.obj.save_iter_db(self.db)

    def save_state_db(self, last_temperature):
        samcroot = self.db.root.samc
        samcroot.theta_hist[:] = self.hist[1,:]
        samcroot.freq_hist[:] = self.hist[2,:].astype(np.int64)

        samcroot._v_attrs.prop_accept = self.accept_loc
        samcroot._v_attrs.prop_total = self.total_loc

        temp = samcroot._v_attrs.temperature
        temp.append(last_temperature) # For some reason this is necessary...
        samcroot._v_attrs.temperature = temp
        samcroot._v_attrs.curr_delta = self.delta

        samcroot._v_attrs.stepscale = self.stepscale
        samcroot._v_attrs.refden_power = self.refden_power
        samcroot._v_attrs.burnin = self.burn
        samcroot._v_attrs.thin = self.thin
        samcroot._v_attrs.curr_iteration = self.iteration

        samcroot._v_attrs.scale = self.scale
        samcroot._v_attrs.grid = self.grid
        samcroot._v_attrs.lowEnergy = self.lowEnergy
        samcroot._v_attrs.highEnergy = self.highEnergy

    def compute_means(self, cummeans=True):
        """ 
        Using the currently saved samples from the object in the pytables db, 
        compute the cumulative mean of the function on the random weighted samples.
        And save the results to the /computed/cummeans region of the db.
        """
        assert self.db != None, 'db not initialized'
        #assert len(self.db) != 0, 'Length of db is zero! Perhaps you have not "\
                #"proceeded beyond the burn-in period'

        thetas = self.db.root.samc.theta_trace.read()
        part = np.exp(thetas - thetas.max())

        if not 'computed' in self.db.root:
            self.db.createGroup('/', 'computed', 'Computed quantities')
        if cummeans and not 'cummeans' in self.db.root.computed:
            cumgroup = self.db.createGroup('/computed', 'cummeans', 'Cumulative means')
        elif cummeans and 'cummeans' in self.db.root.computed:
            cumgroup = self.db.root.computed.cummeans
        if 'means' in self.db.root.computed:
            meangroup = self.db.root.computed.means
        else:
            meangroup = self.db.createGroup('/computed', 'means', 'Means')
        for item in self.db.walkNodes('/object'):
            if isinstance(item, t.array.Array):
                funcs = item.read().astype(np.float)
                if cummeans:
                    numerator = (part * funcs.T).T.cumsum(axis=0)
                    if item.name in cumgroup:
                        raise Exception("Not implemented yet: multiple calls to func_cummean")
                    arr = self.db.createCArray(cumgroup, item.name, 
                            t.Float64Atom(shape=funcs[-1].shape), 
                            (thetas.size,))
                    denom = part.cumsum()
                    arr[:] = (numerator.T / denom).T
                    meangroup._v_attrs[item.name] = arr[-1]
                else:
                    denom = part.sum()
                    numerator = (part * funcs.T).T.sum(axis=0)
                    meangroup._v_attrs[item.name] = (numerator/denom).astype(np.float)

    def truncate_means(self, trunc):
        assert self.db != None, 'db not initialized'
        #assert len(self.db) != 0, 'Length of db is zero! Perhaps you have not "\
                #"proceeded beyond the burn-in period'

        thetas = self.db.root.samc.theta_trace.read()
        n = thetas.shape[0]
        assert 0.0 <= trunc <= 1.0
        last = int(n*(1-trunc))
        part = np.exp(thetas[:last] - thetas[:last].max())
        denom = part.sum()
        meangroup = self.db.createGroup('/computed', 'means_%d'%(int(100*trunc)), 
                'Means truncated at %.2f'%trunc)
        for item in self.db.walkNodes('/object'):
            if isinstance(item, t.array.Array):
                funcs = item.read().astype(np.float)[:last]
                numerator = (part * funcs).sum()
                meangroup._v_attrs[item.name] = float(numerator/denom)

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
    def sample(self, int iters, double temperature = 1.0, object verbose = False):
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
            if verbose and self.iteration % 100 == 0:
                print(self.obj.info())
                #print("""accept:{}, r:{:8.2g}, hist[old]:{:9.2g}, \
                #hist[new]:{:9.2g}, olde:{:6.2g}, newe:{:6.2g} {}/{}""".format(
                    #accept, r, hist[oldregion], hist[newregion], 
                    #oldenergy, newenergy, oldregion,newregion))

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
                self.save_iter_db(hist[oldregion], 
                                oldenergy, 
                                (current_iter-self.burn)//self.thin)

            if self.iteration % 10000 == 0:
                print("Iteration: %8d, delta: %5.2f, best energy: %7g, current energy: %7g" % \
                        (self.iteration, self.delta, self.mapenergy, newenergy))

        self.hist[1] = hist
        self.save_state_db(temperature)

        ###### Calculate summary statistics #######
        print("Accept_loc: %d" % self.accept_loc)
        print("Total_loc: %d" % self.total_loc)
        print("Acceptance: %f" % (float(self.accept_loc)/float(self.total_loc)))


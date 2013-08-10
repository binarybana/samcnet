# cython: profile=True
cimport cython
from libc.math cimport exp, ceil, floor

import sys
import os
import tempfile
import zlib
import tables as t
from collections import Counter

import numpy as np
cimport numpy as np

cdef class MHRun:
    cdef public:
        object obj, db, hist, mapvalue, verbose, scheme_accept, scheme_propose
        int accept_loc, total_loc, iteration, burn, thin
        double mapenergy 
    def __init__(self, obj, burn=100000, thin=1, verbose=False):
        self.verbose = verbose
        self.obj = obj 
        self.clear()

        self.burn = burn
        self.thin = thin
        self.db = None

        self.scheme_accept = Counter()
        self.scheme_propose = Counter()

    def clear(self):
        self.db = None
        self.mapenergy = np.inf
        self.mapvalue = None
        self.iteration = 0
        self.accept_loc = 0
        self.total_loc = 0

    def init_db(self, size):
        if self.db == None:
            filt = t.Filters(complib='bzip2', complevel=7, fletcher32=True)
            if not os.path.exists('.tmp'):
                print("Creating temp directory: .tmp")
                os.mkdir('.tmp')
            name = tempfile.mktemp(prefix='mh', dir='.tmp')
            self.db = t.openFile(name, mode = 'w', title='Metropolis Hastings Run Data', filters=filt)
            self.db.root._v_attrs["mcmc_type"] = 'mh'

            self.db.createGroup('/', 'mh', 'Metropolis Hastings info', filters=filt)
            self.db.createEArray('/mh', 'energy_trace', t.Float64Atom(), (0,), expectedrows=size)

            objdb = self.db.createGroup('/', 'object', 'Object info', filters=filt)
            samples = self.db.createGroup(objdb, 'samples', 'Samples')
            objfxn = self.db.createGroup(objdb, 'objfxn', 'Objective function outputs')

            self.obj.init_db(self.db, self.db.root.object, size)

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

    def save_iter_db(self, double energy, int iteration):
        self.db.root.mh.energy_trace.append((energy,))
        self.obj.save_iter_db(self.db, self.db.root.object)

    def save_state_db(self):
        mhroot = self.db.root.mh

        mhroot._v_attrs.prop_accept = self.accept_loc
        mhroot._v_attrs.prop_total = self.total_loc

        mhroot._v_attrs.burnin = self.burn
        mhroot._v_attrs.thin = self.thin
        mhroot._v_attrs.curr_iteration = self.iteration

    def compute_means(self, cummeans=True):
        """ 
        Using the currently saved samples from the object in the pytables db, 
        compute the cumulative mean of the function on the random weighted samples.
        And save the results to the /computed/cummeans region of the db.
        """
        assert self.db != None, 'db not initialized'
        #assert len(self.db) != 0, 'Length of db is zero! Perhaps you have not "\
                #"proceeded beyond the burn-in period'

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
                    numerator = funcs.cumsum(axis=0)
                    if item.name in cumgroup:
                        raise Exception("Not implemented yet: multiple calls to func_cummean")
                    arr = self.db.createCArray(cumgroup, item.name, 
                            t.Float64Atom(shape=funcs[-1].shape), 
                            (funcs.size,))
                    denom = np.ones(funcs.size).cumsum()
                    arr[:] = (numerator.T / denom).T
                    meangroup._v_attrs[item.name] = arr[-1]
                else:
                    denom = funcs.size()
                    numerator = funcs.sum(axis=0)
                    meangroup._v_attrs[item.name] = (numerator/denom).astype(np.float)

    #@cython.boundscheck(False) # turn off bounds-checking for entire function
    def sample(self, int iters, object verbose = False):
        cdef int current_iter, accept, i, dbsize, scheme
        cdef double oldenergy, newenergy, r
        oldenergy = self.obj.energy()

        dbsize = (self.iteration + int(iters) - self.burn)//self.thin
        if dbsize < 0:
            dbsize = 0
        self.init_db(dbsize)
        
        print("Initial Energy: %g" % oldenergy)

        for current_iter in range(self.iteration, self.iteration + int(iters)):
            self.iteration += 1

            scheme = self.obj.propose()
            newenergy = self.obj.energy()

            if newenergy < self.mapenergy: # NB: Even if not accepted
                self.mapenergy = newenergy
                self.mapvalue = self.obj.copy()
    
            ####### acceptance of new moves #########

            r = oldenergy-newenergy
            
            self.scheme_propose[scheme] += 1
            if np.random.rand() < exp(r):
                accept=1
                self.scheme_accept[scheme] += 1
            else:
                accept=0;
            if verbose:# and self.iteration % 10 == 0:
                print("old: %8.2f, new: %8.2f, r: %5.2f, accept: %d" % (oldenergy, newenergy, r, accept))

            if accept == 0:
                self.obj.reject()
                self.total_loc += 1
            elif accept == 1:
                self.accept_loc += 1
                self.total_loc += 1
                oldenergy = newenergy
                  
            if current_iter >= self.burn and current_iter % self.thin == 0:
                self.save_iter_db(oldenergy, 
                                (current_iter-self.burn)//self.thin)

            if self.iteration % 2000 == 0:
                print("Iteration: %8d, best energy: %7g, current energy: %7g" % \
                        (self.iteration, self.mapenergy, newenergy))

        self.save_state_db()

        ###### Calculate summary statistics #######
        print("Accept_loc: %d" % self.accept_loc)
        print("Total_loc: %d" % self.total_loc)
        print("Acceptance: %f" % (float(self.accept_loc)/float(self.total_loc)))


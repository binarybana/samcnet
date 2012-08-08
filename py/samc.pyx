cimport cython
from libc.math cimport exp, ceil, floor

import tables as tb
import sys
import os
from itertools import izip

import numpy as np
cimport numpy as np

cdef class SAMCRun:
  cdef public:
    object obj, db, refden, hist, indicator, mapvalue
    int lowEnergy, highEnergy, scale, grid, accept_loc, total_loc, iteration, burn, stepscale
    double rho, tau, mapenergy, delta,
  def __init__(self, obj, db='memory'):

    self.obj = obj # Going to be a BayesNet for now, but we'll keep it general
    self.init_db(db)

    self.scale = 1

    self.set_energy_limits()

    self.rho=1.0
    self.tau=1.0;

    self.burn = 100000
    self.stepscale = 100000

  def set_energy_limits(self):
    cdef int i
    cdef double oldenergy, energy, low, high, spread

    if self.iteration > 0:
      print "Reading energy limits from file."
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
      r = oldenergy-energy
      if r > 0.0 or np.random.rand() < exp(r):
        if energy < low:
          low = energy
        elif energy > high:
          high = energy
        oldenergy = energy
      else:
        self.obj.reject()

    spread = high - low
    if spread < 1000:
      spread = 1000
    low = floor(low - .6 * spread)
    high = ceil(high)
    if low < 0:
      low = 0

    print "Done. Setting limits to (%d, %d)" % (low,high)

    self.lowEnergy = <int>low
    self.highEnergy = <int>high
    self.grid = <int>ceil((self.highEnergy - self.lowEnergy) * self.scale)
    self.refden = np.arange(self.grid, 0, -1, dtype=np.double)
    self.refden = self.refden**2
    #self.refden = np.ones(self.grid, dtype=np.double)

    self.refden /= self.refden.sum()
    self.hist = np.zeros((3,self.grid), dtype=np.double)

    self.hist[0,:] = np.arange(self.lowEnergy, self.highEnergy, 1./self.scale)
    self.indicator = np.zeros((self.grid),dtype=np.int32) # Indicator is whether we have visited a region yet

  def clear(self):
    if type(self.db) == list:
      self.db = []
      self.init_db('memory')
    else:
      try:
        fname = self.db.filename
        self.db.close()
        print "Removing database file %s." % fname
        os.remove(fname)
      except OSError:
        print "Error cleaning up database, cannot delete file %s." % fname
      finally:
        self.init_db(fname)

  def init_db(self,db):
    self.mapenergy = np.inf
    self.mapvalue = None
    self.delta = 1.0
    self.iteration = 0
    self.accept_loc = 0
    self.total_loc = 0
    if db == 'memory':
      self.db = [] # Place to store the samples
    else: 
      self.db = tb.openFile(db, mode='a', title='SAMCRun', 
          filters = tb.Filters(complevel=4, complib='blosc'))
      if '/samples' in self.db and len(self.db.root.samples) > 0:
        self.delta = self.db.getNodeAttr('/','delta')
        self.mapenergy = self.db.getNodeAttr('/','mapenergy')
        self.mapvalue = self.db.getNodeAttr('/','mapvalue')
        self.iteration = self.db.getNodeAttr('/','iteration')
        self.accept_loc = self.db.getNodeAttr('/','accept_loc')
        self.total_loc = self.db.getNodeAttr('/','total_loc')
        self.hist = self.db.getNodeAttr('/','hist')
        self.grid = self.db.getNodeAttr('/','grid')
        self.indicator = self.db.getNodeAttr('/','indicator')
        self.refden = self.db.getNodeAttr('/','refden')

      self.obj.init_db(self.db) # Only for pytables dbs

  def estimate_func_mean(self, func):
    """ Using the supplied function, estimate the mean of the function
    on the random samples.

    It might be best to sort the values first for numerical stability,
    but we'll worry about that later.
    """
    if type(self.db) == list:
      thetas = np.array([x['theta'] for x in self.db])
      nets = self.db
    else:
      thetas = self.db.root.samples[:]['theta']
      nets = self.db.root.samples[:]

    part = np.exp(thetas - thetas.max())
    numerator = (part * np.array([func(x) for x in nets])).sum()
    denom = part.sum()
    print "Calculating function mean: %g / %g." % (numerator, denom)
    return numerator / denom

  def save_attribs(self):
    if type(self.db) != list:
      self.db.setNodeAttr('/','delta',self.delta)
      self.db.setNodeAttr('/','mapenergy',self.mapenergy)
      self.db.setNodeAttr('/','mapvalue',self.mapvalue)
      self.db.setNodeAttr('/','iteration',self.iteration)
      self.db.setNodeAttr('/','accept_loc',self.accept_loc)
      self.db.setNodeAttr('/','total_loc',self.total_loc)
      self.db.setNodeAttr('/','hist',self.hist)
      self.db.setNodeAttr('/','grid',self.grid)
      self.db.setNodeAttr('/','indicator',self.indicator)
      self.db.setNodeAttr('/','refden',self.refden)
      self.db.flush()

  def __del__(self):
    self.save_attribs()

  cdef find_region(self, energy):
    if energy > self.highEnergy: 
      return self.grid-1
    elif energy < self.lowEnergy:
      return 0
    else: 
      return floor((energy-self.lowEnergy)*self.scale)

  @cython.boundscheck(False) # turn off bounds-checking for entire function
  def sample(self, iters, thin=1):
    cdef int current_iter, accept, oldregion, newregion, i, nonempty
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

    print("Initial Energy: %g" % oldenergy)
    #fid = open("rlogpy2",'w')

    try:
      for current_iter in range(self.iteration, self.iteration + int(iters)):
        self.iteration += 1

        self.delta = float(self.stepscale) / max(self.stepscale, self.iteration)

        self.obj.propose()
        newenergy = self.obj.energy()

        if newenergy < self.mapenergy: # NB: Even if not accepted
          self.mapenergy = newenergy
          self.mapvalue = self.obj.copy()
    
        ####### acceptance of new moves #########

        newregion = self.find_region(newenergy)

        indicator[newregion] = 1

        r = hist[oldregion] - hist[newregion] + (oldenergy-newenergy) #/self.temperature
        
        #fid.write("%f,%f,%f,%f,%f,%f\n" % (hist[oldregion], hist[newregion], oldenergy,
          #newenergy, r, self.obj.lastscheme))
        
        #print("r:%f\t oldregion:%d\t hist[old]:%f\t hist[new]:%f fold:%f, fnew:%f" %
            #(r,oldregion, hist[1,oldregion], hist[1,newregion], oldenergy, newenergy))

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

        self.obj.save_to_db(self.db, oldenergy, hist[oldregion], oldregion)

        # VV I can't see what purpose this would serve from the paper (except
        # maybe the varying truncation?)
        #if current_iter == self.burn:
          #i = 0
          #while i<=self.grid and indicator[i] == 0:
            #i+=1
          #nonempty = i
        #elif current_iter > self.burn: # FIXME BuG here as nonempty is not persisted
          #un = hist[nonempty]
          #for i in range(self.grid):
            #if indicator[i] == 1:
              #hist[i] -= un
              
        if self.iteration % 10000 == 0:
          print("Iteration: %8d, delta: %5.2f, best energy: %7g, current energy: %7g" % \
              (self.iteration, self.delta, self.mapenergy, newenergy))
    except KeyboardInterrupt: 
      pass
    self.hist[1] = hist
    self.indicator = indicator

    ###### Calculate summary statistics #######
    print("Accept_loc: %d" % self.accept_loc)
    print("Total_loc: %d" % self.total_loc)
    print("Acceptance: %f" % (float(self.accept_loc)/float(self.total_loc)))

    self.save_attribs()


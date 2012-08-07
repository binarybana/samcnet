import pylab as p
import os
import networkx as nx
import numpy as np
import pandas as pa
import pebl as pb
import StringIO as si
import tempfile

def plotHist(s):
    p.subplot(211)
    p.plot(s.hist[0], s.hist[1], 'go')
    p.title('Energy - theta')
    p.subplot(212)
    p.plot(s.hist[0], s.hist[2], 'go')
    p.title('Sample Counts')

def plotThetas(s):
  self = s
  if type(self.db) == list:
    thetas = np.array([x['theta'] for x in self.db])
    nets = self.db
  else:
    thetas = self.db.root.samples[:]['theta']
    nets = self.db.root.samples[:]
  part = np.exp(thetas - thetas.max())
  p.hist(part, log=True, bins=100)
    
def drawGraph(graph, show=False):
    fname = os.tempnam()
    nx.write_dot(graph, fname+'.dot')
    os.popen('dot -Tsvg -o %s.svg %s.dot' % (fname,fname))
    if show:
        os.popen('xdg-open %s.svg > /dev/null' % fname)
    return fname

def drawGraphs(*args, **kwargs):
    agraphs = [nx.to_agraph(graph) for graph in args] 
    
    files = [tempfile.mkstemp(suffix='.svg') for x in agraphs]
    for f in files:
        os.close(f[0])

    agraphs[0].layout(prog='dot')
    agraphs[0].draw(files[0][1])
    agraphs[0].remove_edges_from(agraphs[0].edges())

    for fname,g in zip(files[1:],agraphs[1:]):
        agraphs[0].add_edges_from(g.edges())
        agraphs[0].draw(fname[1])
        agraphs[0].remove_edges_from(g.edges())

    combo = tempfile.mkstemp(suffix='.png')
    os.close(combo[0])
    os.popen('convert %s +append -quality 75 %s' % (' '.join(zip(*files)[1]), combo[1]))
    if 'show' in kwargs and not kwargs['show']:
      pass
    else:
      os.popen('xdg-open %s > /dev/null' % combo[1])

    for f in files:
        os.unlink(f[1])

def to_pebl(states, data):
    header = ['%d,discrete(%d)' %(i,a) for i,a in enumerate(states)]
    df = pa.DataFrame(data, columns=header)
    x = si.StringIO()
    x.write('\t'.join(header) + '\n')
    df.to_csv(x, header=False, index=True, sep='\t')
    x.seek(0)
    return pb.data.fromstring(x.read())

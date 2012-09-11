import pylab as p
import os
import networkx as nx
import numpy as np
import pandas as pa
#import pebl as pb
import StringIO as si
import tempfile

def plotHist(s):
    rows = 3
    cols = 2

    p.figure()
    p.subplot(rows, cols, 1)
    p.plot(s.hist[0], s.hist[1], 'k.')
    p.title("Region's theta values")
    p.ylabel('Theta')
    p.xlabel('Energy')

    p.subplot(rows, cols, 2)
    p.plot(s.hist[0], s.hist[2], 'k.')
    p.title("Region's Sample Counts")
    p.ylabel('Count')
    p.xlabel('Energy')

    energies = s.db['energies']
    thetas = s.db['thetas']

    p.subplot(rows, cols, 3)
    p.plot(np.arange(s.burn, energies.shape[0]+s.burn), energies, 'k.')
    p.title("Energy Trace")
    p.ylabel('Energy')
    p.xlabel('Iteration')

    p.subplot(rows, cols, 4)
    p.plot(np.arange(s.burn, thetas.shape[0]+s.burn), thetas, 'k.')
    p.ylabel('Theta Trace')
    p.xlabel('Iteration')
        
    p.subplot(rows, cols, 5)
    part = np.exp(thetas - thetas.max())
    p.hist(part, log=True, bins=100)
    p.xlabel('exp(theta - theta_max)')
    p.ylabel('Number of samples at this value')
    p.title('Histogram of normalized sample thetas from %d iterations' % thetas.shape[0])

    p.subplot(rows, cols, 6)
    p.hist(part, weights=part, bins=50)
    p.xlabel('exp(theta - theta_max)')
    p.ylabel('Amount of weight at this value')

def plotScatter(s):
    energies = s.db['energies']
    thetas = s.db['thetas']

    p.figure()
    p.plot(energies, thetas, 'k.', alpha=0.7)
    p.xlabel('Energy')
    p.ylabel('Theta')

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

import pylab as p
import os
import networkx as nx
import numpy as np
import pandas as pa
#import pebl as pb
import StringIO as si
import tempfile
import tables as t

from probability import CPD,fast_space_iterator,JointDistribution

def graph_to_joint(graph):
    joint = JointDistribution()
    cpds = []
    for node in graph.nodes():
        marg = graph.node[node]['marginal']
        eta = graph.node[node]['eta']
        delta = graph.node[node]['delta']
        if np.isnan(marg): # yes parents
            params = {(0,):np.r_[1-eta], (1,):np.r_[delta]}
            pars = {graph.predecessors(node)[0]:2}
        else:
            params = {():np.r_[marg]}
            pars = {}
        joint.add_distribution(CPD(node,2,params,pars))

    return joint

def getHost():
    return os.uname()[1].split('.')[0]

def plot_h5(loc):
    fid = t.openFile(loc, 'r')
    samcattrs = fid.root.samc._v_attrs
    energy = np.linspace(samcattrs['lowEnergy'], samcattrs['highEnergy'], samcattrs['grid'])
    theta = fid.root.samc.theta_hist.read()
    counts = fid.root.samc.freq_hist.read()
    theta_trace = fid.root.samc.theta_trace.read()
    energy_trace = fid.root.samc.energy_trace.read()
    burn = samcattrs['burnin']
    _plot_SAMC(energy, theta, counts, energy_trace, theta_trace, burn)

def plotHist(s):
    _plot_SAMC(s.hist[0], s.hist[1], s.hist[2],
            s.db.root.samc.energy_trace.read(),  
            s.db.root.samc.theta_trace.read(), s.burn)

def _plot_SAMC(energy, theta, counts, energy_trace, theta_trace, burn):
    rows = 3
    cols = 2

    p.figure()
    p.subplot(rows, cols, 1)
    p.plot(energy, theta, 'k.')
    p.title("Region's theta values")
    p.ylabel('Theta')
    p.xlabel('Energy')

    p.subplot(rows, cols, 2)
    p.plot(energy, counts, 'k.')
    p.title("Region's Sample Counts")
    p.ylabel('Count')
    p.xlabel('Energy')

    p.subplot(rows, cols, 3)
    p.plot(np.arange(burn, energy_trace.shape[0]+burn), energy_trace, 'k.')
    p.title("Energy Trace")
    p.ylabel('Energy')
    p.xlabel('Iteration')

    p.subplot(rows, cols, 4)
    p.plot(np.arange(burn, theta_trace.shape[0]+burn), theta_trace, 'k.')
    p.ylabel('Theta Trace')
    p.xlabel('Iteration')
        
    p.subplot(rows, cols, 5)
    part = np.exp(theta_trace - theta_trace.max())
    p.hist(part, log=True, bins=100)
    p.xlabel('exp(theta - theta_max)')
    p.ylabel('Number of samples at this value')
    p.title('Histogram of normalized sample thetas from %d iterations' % theta_trace.shape[0])

    p.subplot(rows, cols, 6)
    p.hist(part, weights=part, bins=50)
    p.xlabel('exp(theta - theta_max)')
    p.ylabel('Amount of weight at this value')

def plot_nodes(loc, node, parts=[0.0, 0.1, 0.2]):
    filelist = os.listdir(loc)
    n = len(filelist)
    avgs = [np.empty(n) for i in parts]
    for i,d in enumerate(filelist):
        fid = t.openFile(os.path.join(loc,d), 'r')
        theta_trace = fid.root.samc.theta_trace.read()
        n = theta_trace.shape[0]
        array = fid.getNode(node).read()
        inds = theta_trace.argsort()
        theta_sort = theta_trace[inds]
        array_sort = array[inds]

        for j,frac in enumerate(parts):
            last = int(n*(1-frac))
            part = np.exp(theta_sort[:last] - theta_sort[:last].max())
            denom = part.sum()
            numerator = (part * array_sort[:last]).sum()
            avgs[j][i] = numerator / denom
    # Plotting
    rows = len(parts) 
    cols = 1
    agg = np.hstack(avgs)
    bins = np.linspace(agg.min()-0.3, agg.max()+0.3, 20)
    p.figure()
    for i,frac in enumerate(parts):
        p.subplot(rows, cols, i+1)
        p.hist(avgs[i], bins=bins)
        p.title('%s at %.3f fraction' % (node, frac))

def plot_thetas(loc, parts=[0.1, 0.2]):
    fid = t.openFile(loc, 'r')
    theta_trace = fid.root.samc.theta_trace.read()
    _plot_thetas(theta_trace, parts)

def _plot_thetas(theta_trace, parts):
    rows = len(parts) + 1
    cols = 2

    theta_trace.sort()
    n = theta_trace.shape[0]

    def plot_theta_hist(i, frac):
        p.subplot(rows, cols, 2*i+1)
        last = int(n*(1-frac))
        part = np.exp(theta_trace[:last] - theta_trace[:last].max())
        p.hist(part, log=True, bins=100)
        p.xlabel('exp(theta - theta_max)')
        p.ylabel('Number of samples at this value')
        p.title('Normalized sample thetas at %.3f' % frac)

        p.subplot(rows, cols, 2*i+2)
        p.hist(part, weights=part, bins=50)
        p.xlabel('exp(theta - theta_max)')
        p.ylabel('Amount of weight at this value')

    p.figure()
    for i,part in enumerate([0.0] + parts):
        plot_theta_hist(i,part)

def plotScatter(s):
    energies = s.db.root.samc.energy_trace.read()
    thetas = s.db.root.samc.theta_trace.read()

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

def best_to_graph(mapvalue):
    mat = mapvalue[0]
    x = mapvalue[1]
    s = x.argsort()
    mat = mat[s].T[s].T
    np.fill_diagonal(mat, 0)
    return nx.from_numpy_matrix(mat)

def to_pebl(states, data):
    header = ['%d,discrete(%d)' %(i,a) for i,a in enumerate(states)]
    df = pa.DataFrame(data, columns=header)
    x = si.StringIO()
    x.write('\t'.join(header) + '\n')
    df.to_csv(x, header=False, index=True, sep='\t')
    x.seek(0)
    return pb.data.fromstring(x.read())

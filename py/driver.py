if __name__ == '__channelexec__' or __name__=='__main__':
    import sys, os, io
    import numpy as np
    import networkx as nx
    import ConfigParser as cp

    from time import time

    sys.path.append('../build')
    sys.path.append('./build') # Yuck!

    from samc import SAMCRun
    from bayesnet import BayesNet
    from generator import *

    #Just for now...
    test = """
[General]
nodes = 5
samc-iters=5e5
numdata=50
priorweight=10
numtemplate=5
    """
    os.environ['SAMC_CONFIG'] = test
    print test.encode('ascii')

    config = cp.RawConfigParser()
    config.readfp(io.BytesIO(os.environ['SAMC_CONFIG']))

    N = config.getfloat('General', 'nodes')
    iters = config.getfloat('General', 'samc-iters')
    numdata = config.getint('General', 'numdata')
    priorweight = config.getfloat('General', 'priorweight')
    numtemplate = config.getint('General', 'numtemplate')
    #db = config.get('General', 'db')

    graph = generateHourGlassGraph(nodes=N)
    gmat = np.asarray(nx.to_numpy_matrix(graph))

    def global_edge_presence(net):
        s = net['x'].argsort()
        ordmat = net['matrix'][s].T[s].T
        return np.abs(gmat - ordmat).sum() / net['x'].shape[0]**2

    template = sampleTemplate(graph, numtemplate)
    tmat = np.asarray(nx.to_numpy_matrix(template))
    traindata, states, cpds = generateData(graph,numdata)
    nodes = np.arange(graph.number_of_nodes())

    nodes = np.arange(traindata.shape[1])
    b = BayesNet(nodes,states,traindata,template=tmat)
    s = SAMCRun(b)

    t1 = time()
    s.sample(iters)
    t2 = time()
    print("SAMC run took %f seconds." % (t2-t1))
    func_mean = s.estimate_func_mean(global_edge_presence)
    t3 = time()
    print("Mean estimation run took %f seconds." % (t3-t2))

    # Send back func_mean to store
    if __name__ == '__channelexec__':
      channel.send(float(func_mean))
    else:
      print(func_mean)


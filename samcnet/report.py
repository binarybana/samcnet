import os, sys, zlib, redis
import pylab as p
import numpy as np
import tables as t

def h5_plot(ax, node, filelist, TMPDIR, ylabel=None):
    first = True
    for d in filelist:
        fid = t.openFile(os.path.join(TMPDIR, d), 'r')
        obj = fid.getNode(node)
        label = obj.name
        if label == 'freq_hist':
            x = np.linspace(fid.root.samc._v_attrs['lowEnergy'],
                    fid.root.samc._v_attrs['highEnergy'], 
                    fid.root.samc._v_attrs['grid'])
        else:
            x = np.arange(obj.read().size)
        if first:
            ax.plot(x, obj.read(), 'b', alpha=0.4, label=label)
        else:
            ax.plot(x, obj.read(), 'b', alpha=0.4)
        first = False
        if 'descs' in fid.root.object._v_attrs and label in fid.root.object._v_attrs.descs:
            if ylabel:
                p.ylabel(ylabel)
            else:
                p.ylabel(fid.root.object._v_attrs.descs[label])
        fid.close()
    ax.grid(True)
    ax.legend()

if __name__ == "__main__":
    r = redis.StrictRedis('localhost')
    TMPBASE = '/tmp/samcfiles'
    if not os.path.exists(TMPBASE):
        os.mkdir('/tmp/samcfiles')

    # Get all job hashes with results and sort by time submitted
    done_hashes = sorted(r.keys('jobs:done:*'), key=lambda x: int(r.hget('jobs:times', x[10:]) or '0'))

    # Print results
    for i, d in enumerate(done_hashes):
        desc = r.hget('jobs:descs', d[10:]) or ''
        num = r.llen(d)
        print "%4d. (%3s) %s %s" % (i, num, d[10:15], desc)

    sel = raw_input("Choose a dataset or 'q' to exit: ")
    if not sel.isdigit() or int(sel) not in range(i+1):
        sys.exit()

    sel = int(sel)
    fulljobhash = done_hashes[sel]
    jobhash = fulljobhash[10:]

    # Test if we already have pulled this data down
    TMPDIR = os.path.join(TMPBASE, jobhash)
    if not os.path.isdir(TMPDIR):
        os.mkdir(TMPDIR)

    filelist = os.listdir(TMPDIR)
    if len(filelist) != r.llen(fulljobhash):
        # Grab datasets
        datastrings = r.lrange(fulljobhash, 0, -1)
        print "Persisting %d datasets from hash %s" % (len(datastrings), jobhash[:5])
        for i,data in enumerate(datastrings):
            with open(os.path.join(TMPDIR, str(i)), 'w') as fid:
                fid.write(zlib.decompress(data))
        filelist = os.listdir(TMPDIR)
    else:
        print "Found %d datasets from hash %s in cache" % (len(os.listdir(TMPDIR)), jobhash[:5])

    #obj = fid.root.samc.theta_trace
    #obj2 = fid.root.samc.energy_trace
    #obj = fid.root.samc.freq_hist
    #obj = fid.root.samc.energy_trace
    #obj = fid.root.object.objfxn.entropy
    #obj = fid.root.object.objfxn.edge_distance
    #obj = fid.root.computed.cummeans.kld
    #obj = fid.root.computed.cummeans.entropy
    #obj = fid.root.computed.cummeans.edge_distance

    plot_list = [
                #'/samc/freq_hist', 
                '/computed/cummeans/entropy', 
                '/computed/cummeans/kld', 
                '/computed/cummeans/edge_distance']
    label_list = [
                #'Samples from energy',
                'Entropy in bits',
                'KLD in bits',
                'Incorrect edge proportion']
    p.figure()
    for i,node in enumerate(plot_list):
        h5_plot(p.subplot(len(plot_list), 1, i+1), node, filelist, TMPDIR, label_list[i])
        if i==0:
            p.title(r.hget('jobs:descs', jobhash) + "\n" + \
                    'Experiment version: ' + jobhash[:5] + '\n' + \
                    'Code version: ' + r.hget('jobs:githashes', jobhash)[:5])

    # Grab info that should be identical for all samples
    fid = t.openFile(os.path.join(TMPDIR, '0'), 'r')
    print("###### SAMC ######")
    for name in fid.root.samc._v_attrs._f_list('user'):
        print("%30s:\t%s" % (name, str(fid.root.samc._v_attrs[name])))
    print("###### Object ######")
    for name in fid.root.object._v_attrs._f_list('user'):
        print("%30s:\t%s" % (name, str(fid.root.object._v_attrs[name])))
    fid.close()

    if True:
        import cPickle as cp
        import networkx as nx
        import subprocess as sb
        x = r.hget('jobs:grounds', jobhash)
        z = cp.loads(zlib.decompress(x))
        nx.write_dot(z, '/tmp/tmp.dot')
        sb.call('dot /tmp/tmp.dot -Tpng -o /tmp/tmp.png'.split())

    p.xlabel('Samples obtained after burnin (after thinning)')

    p.show()

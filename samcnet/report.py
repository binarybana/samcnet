import os, sys, zlib, redis
import pylab as p
import numpy as np
import tables as t

def sweep_plot(r, jobhash):
    p.figure()
    keystart = 'custom:%s:trunc=10:samplesize=' % jobhash
    keys = r.keys(keystart + '*')
    cut = len(keystart)
    #filter(lambda x: x.startswith('samplesize'), s.split(':'))
    positions = [int(x[cut:]) for x in keys] 
    datasets = [map(float,r.lrange(x,0,-1)) for x in keys]

    p.boxplot(datasets, positions=positions, widths=4)
    p.grid(True)
    p.xlim(0,170)
    #p.legend()

def cummeans_plot(ax, filelist, node, ylabel=None):
    first = True
    for d in filelist:
        fid = t.openFile(d, 'r')
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

def prompt_for_dataset(r):
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
    return done_hashes[int(sel)][10:]

def pull_data(r, basedir, jobhash):
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    # Test if we already have pulled this data down
    TMPDIR = os.path.join(basedir, jobhash)
    if not os.path.isdir(TMPDIR):
        os.mkdir(TMPDIR)
    filelist = os.listdir(TMPDIR)
    if len(filelist) != r.llen('jobs:done:'+jobhash):
        # Grab datasets
        datastrings = r.lrange('jobs:done:'+jobhash, 0, -1)
        print "Persisting %d datasets from hash %s" % (len(datastrings), jobhash[:5])
        for i,data in enumerate(datastrings):
            with open(os.path.join(TMPDIR, str(i)), 'w') as fid:
                fid.write(zlib.decompress(data))
        filelist = os.listdir(TMPDIR)
    else:
        print "Found %d datasets from hash %s in cache" % (len(os.listdir(TMPDIR)), jobhash[:5])
    return [os.path.join(TMPDIR, x) for x in filelist]

def print_h5_info(loc):
    # Grab info that should be identical for all samples
    fid = t.openFile(loc, 'r')
    print("###### SAMC ######")
    for name in fid.root.samc._v_attrs._f_list('user'):
        print("%30s:\t%s" % (name, str(fid.root.samc._v_attrs[name])))
    print("###### Object ######")
    for name in fid.root.object._v_attrs._f_list('user'):
        print("%30s:\t%s" % (name, str(fid.root.object._v_attrs[name])))
    fid.close()

def show_ground(r, jobhash):
    import cPickle as cp
    import networkx as nx
    import subprocess as sb
    x = r.hget('jobs:grounds', jobhash)
    z = cp.loads(zlib.decompress(x))
    nx.write_dot(z, '/tmp/tmp.dot')
    sb.call('dot /tmp/tmp.dot -Tpng -o /tmp/tmp.png'.split())
    sb.call('xdg-open /tmp/tmp.png'.split())

if __name__ == "__main__":

    r = redis.StrictRedis('localhost')
    jobhash = prompt_for_dataset(r)

    #basedir = '/tmp/samcfiles'
    #filelist = pull_data(r, basedir, jobhash)

    #plot_list = [
                ##'/samc/freq_hist', 
                #'/computed/cummeans/entropy', 
                #'/computed/cummeans/kld', 
                #'/computed/cummeans/edge_distance']
    #label_list = [
                ##'Samples from energy',
                #'Entropy in bits',
                #'KLD in bits',
                #'Incorrect edge proportion']
    #p.figure()
    #for i,node in enumerate(plot_list):
        #cummeans_plot(p.subplot(len(plot_list), 1, i+1), filelist, node, label_list[i])
        #if i==0:
            #p.title(r.hget('jobs:descs', jobhash) + "\n" + \
                    #'Experiment version: ' + jobhash[:5] + '\n' + \
                    #'Code version: ' + r.hget('jobs:githashes', jobhash)[:5])

    #p.xlabel('Samples obtained after burnin (after thinning)')

    #print_h5_info(filelist[0])

    sweep_plot(r, jobhash)

    p.show()

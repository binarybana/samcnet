import os, sys, zlib, redis
import pylab as p
from matplotlib import rc
import numpy as np
import tables as t

rc('text', usetex=True)

#custom:3dabbb5ffad97a8d205a6b22bd9543f5c5a0e1b7:p_struct=15:ntemplate=4:p_cpd=0

def sweep_plot(r, jobhashes):
    def transform(s):
        return dict([x.split('=') for x in s.split(':')])
    def filt(crit, params, datasets):
        for i,p in enumerate(params):
            cp = dict([(k,int(v)) for k,v in p.iteritems()])
            if cp == crit:
                return i
    params = []
    datasets = []
    for jobhash in jobhashes:
        keystart = 'custom:%s:' % jobhash
        cutlen = len(keystart)
        keys = r.keys(keystart + '*')

        params.extend([transform(x[cutlen:]) for x in keys])
        datasets.extend([map(float,r.lrange(x,0,-1)) for x in keys])

    pen = ['g','r','m','c']
    x = [0.5,1.5,3.0,5.0,8.0,15.0,30.0]
    for pristine_cpd in [0, 1]:
        for numtemplate in [4,8]:
            res = []
            err = []
            for p_struct in map(lambda t: int(t*10), x):
                cpd = pristine_cpd * p_struct
                ind = filt({'ntemplate':numtemplate,
                        'p_cpd':cpd,
                        'p_struct':p_struct},
                        params, datasets)
                res.append(np.median(datasets[ind]))
                err.append(np.std(datasets[ind]))
            p.errorbar(x,res,yerr=err,
                    color=pen.pop(), 
                    label='%s; $\,$ %d edges' % 
                        (r'$\gamma_{\textrm{cpd}}=\gamma_{\textrm{structural}}$' if pristine_cpd else 
                        r'$\gamma_{\textrm{cpd}}=0$',
                        numtemplate),
                    linewidth=2)
    
    #p.boxplot(datasets, positions=[int(x['p_struct'])/10. for x in params])

    p.grid(True)
    p.legend(bbox_to_anchor=(.5,1), loc=2, fontsize=12)
    #legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=gcf().transFigure)
    p.xlim(0,31)
    p.xlabel(r'$\gamma_{\textrm{structural}}$')#,fontsize=20)
    p.ylabel('Posterior average of KLD')
    #p.ylim(0,5)

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

def prompt_for_dataset(r, n=[]):
    # Get all job hashes with results and sort by time submitted
    done_hashes = sorted(r.keys('jobs:done:*'), key=lambda x: int(r.hget('jobs:times', x[10:]) or '0'))
    if n:
        return [done_hashes[x][10:] for x in n]
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
    #jobhash = prompt_for_dataset(r)

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

    jobhashes = prompt_for_dataset(r, [25,26,27,28,29,30,31])
    show_ground(r, jobhashes[0])
    sweep_plot(r, jobhashes)

    p.show()

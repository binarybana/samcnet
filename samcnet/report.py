import os, sys, zlib, redis
import pylab as p
import numpy as np
import tables as t

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

first = True
for d in filelist:
    fid = t.openFile(os.path.join(TMPDIR, d), 'r')

    #obj = fid.root.samc.theta_trace
    #obj2 = fid.root.samc.energy_trace
    #obj = fid.root.samc.freq_hist
    #obj = fid.root.samc.energy_trace
    #obj = fid.root.object.objfxn.entropy
    #obj = fid.root.object.objfxn.edge_distance
    obj = fid.root.computed.cummeans.kld
    #obj = fid.root.computed.cummeans.entropy
    #obj = fid.root.computed.cummeans.edge_distance
    label = obj.name
    #p.plot(obj.read(), 'b', alpha=0.4)
    #p.plot(obj.read(), obj2.read(), 'b.', alpha=0.4)
    if first:
        p.plot(obj.read(), 'b', alpha=0.4, label=label)
    else:
        p.plot(obj.read(), 'b', alpha=0.4)
    first = False

    fid.close()

    #n = len(res)
    #indices = np.linspace(0,1,n)
    #for i,val in enumerate(res):
        #if type(val) == np.ndarray:
            #p.plot(val, alpha=0.4, color='blue')#color=p.cm.jet(indices[i]))
            ##print indices[i], p.cm.jet(indices[i])
        #else:
            #print val

# Grab info that should be identical for all samples
fid = t.openFile(os.path.join(TMPDIR, d), 'r')
print("###### SAMC ######")
for name in fid.root.samc._v_attrs._f_list('user'):
    print("%30s:\t%s" % (name, str(fid.root.samc._v_attrs[name])))
print("###### Object ######")
for name in fid.root.object._v_attrs._f_list('user'):
    print("%30s:\t%s" % (name, str(fid.root.object._v_attrs[name])))
if 'descs' in fid.root.object._v_attrs and label in fid.root.object._v_attrs.descs:
    p.ylabel(fid.root.object._v_attrs.descs[label])
fid.close()
p.xlabel('Samples obtained after burnin (after thinning)')
p.title(r.hget('jobs:descs', jobhash) + "\n" + \
        'Experiment version: ' + jobhash[:5] + '\n' + \
        'Code version: ' + r.hget('jobs:githashes', jobhash)[:5])
p.grid(True)
p.legend()

p.show()

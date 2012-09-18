#!/usr/bin/env python
import os, sys, shlex, time, sha
import subprocess as sb
import redis
import simplejson as js
from samcnet.server_configs import serverconfigs, syncgroups, cesg_small, cesg_large, gsp_compute, gsp_compute_all

LocalRoot = '/home/bana/GSP/research/samc/code'

def launchClient(host):
    cores = host.cores
    if host.cde:
        spec = ('ssh {0.hostname} cd {0.root}/cde-package/cde-root/' \
                + 'home/bana/GSP/research/samc/code; ').format(host) \
                + ('nohup {0.python} -m samcnet.driver {1} >/dev/null 2>&1 &'.format(host,cores))
    else:
        spec = ('ssh {0.hostname} cd {0.root};' + \
                'LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH nohup {0.python} '+ \
                '-m samcnet.driver {1} >/dev/null 2>&1 &').format(host,cores)

    print "Connecting to %s." % host.hostname
    p = sb.Popen(shlex.split(spec), 
            bufsize=-1)
            #stdout=open('/tmp/samc-{0.hostname}-{1}.log'.format(host, random.randint(0,1e9)) ,'w'))
    return 

def manualKill(host):
    print 'Killing processes on %s.' % host.hostname
    user = host.root.split('/')[2]
    spec = 'ssh {0.hostname} killall -q -u {1} python; killall -q -u {1} python2.7; killall -q -u {1} cde-exec'.format(host, user)
    sb.Popen(shlex.split(spec))

def sync(group):
    if group.cde:
        print ("Beginning cde rsync to %s... " % group.hostname)
        p = sb.Popen('rsync -acz {0}/cde-package {1.hostname}:{1.dir}'.format(LocalRoot, group).split())
    else:
        print ("Beginning code rsync to %s... " % group.hostname)
        p = sb.Popen('rsync -acz --exclude=*cde* --exclude=lib --exclude=.lock* --exclude=build {0}/ {1.hostname}:{1.dir}/'.format(LocalRoot, group).split())
        print ' Done.'
        p.wait()
        print ("Beginning remote build to %s... " % group.hostname)
        p = sb.Popen(shlex.split('ssh {0.hostname} "cd {0.dir}; ./waf distclean; . cfg; ./waf"'.format(group)))
    p.wait()
    print ' Done.'

def updateCDE():
    print "Updating CDE package..." 
    os.environ['LD_LIBRARY_PATH']='build:lib'
    os.environ['PYTHONPATH']=LocalRoot
    os.chdir(LocalRoot)
    #p = sb.Popen('/home/bana/bin/cde python {0}/samcnet/driver.py rebuild'.format(LocalRoot).split())
    p = sb.Popen('/home/bana/bin/cde python -m samcnet.driver rebuild'.split())
    p.wait()
    p = sb.Popen('rsync -a samcnet cde-package/cde-root/home/bana/GSP/research/samc/code/'.format(LocalRoot).split())
    p.wait()
    p = sb.Popen('rsync -a lib cde-package/cde-root/home/bana/GSP/research/samc/code/'.split())
    p.wait()
    p = sb.Popen('rsync -a build cde-package/cde-root/home/bana/GSP/research/samc/code/'.split())
    p.wait()
    print " Done."

def postJob(job, samples, single=False):
    """
    Take a dictionary with a minimum of the following keys defined (values are just examples):
            nodes = 5,
            samc_iters=1e4,
            numdata=50,
            priorweight=10,
            numtemplate=5)
    and post a desired <samples> number of runs to be performed.
    """
    jsonjob = js.dumps(job)
    h = sha.sha(jsonjob).hexdigest()
    r.hsetnx('configs', h, jsonjob)
    if single:
        r.hsetnx('single-configs', h, jsonjob)
    tot = r.hincrby('desired-samples', h, samples)
    print("Added %d samples for a total of %d samples remaining." % (samples,tot))
    print("Pushed job: hash %s" % h[:8])
    for k,v in job.iteritems():
        if k == 'graph' or k == 'joint':
            print '   {0:<20} {1:<30}'.format(k, str(sha.sha(v).hexdigest())[:8])
        else:
            print '   {0:<20} {1:<30}'.format(k, str(v))

def postSweep(base, iters, param, values):
    """ 
    Take the <base> config, and get <iters> samples across the <values> in <param>.
    Also save the base config in 'sweep-configs' in Redis.
    """
    assert param in base
    base[param] = 'sweep'
    for k,v in base.iteritems():
        if k != param and v == 'sweep':
            raise Exception('Improper configuration specification.')
    sweepconfig = js.dumps(base)
    sweephash = sha.sha(sweepconfig).hexdigest()
    r.hsetnx('sweep-configs', sweephash, sweepconfig)
    for v in values:
        base[param] = v
        r.hsetnx('sweep-'+sweephash, v, sha.sha(js.dumps(base)).hexdigest())
        postJob(base, iters)

def kill(target):
    assert not r.exists('die')
    if r.zcard('clients-hb') == 0:
        print 'No living clients to kill.'
        return
    print "Sending out kill command to %s." % target

    def countTargets(target):
        if target == 'all':
            return r.zcard('clients-hb')
        else:
            clients = r.sort('clients-hb', by='nosort')
            return len([x for x in clients if x == target])

    num = countTargets(target)
    print ('Waiting for %s clients to die...' % num)
    r.set('die', target)

    try:
        while countTargets(target)>0:
            time.sleep(1)
        print("%d clients killed." % num)
    except KeyboardInterrupt:
        pass
    finally:
        r.delete('die')

def getGraph(N, intercon = 2):
        from samcnet.utils import drawGraphs
        import networkx as nx
        import numpy as np
        from samcnet.generator import generateHourGlassGraph
        cont = 'n'
        while cont != 'y':
            g = generateHourGlassGraph(N, intercon)
            drawGraphs(g)
            cont = raw_input('Is this graph okay? (y/n): ')
        return g


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: python exec.py [sync <groupname>] [syncall]"+\
                " [postdummy] [kill <hostname>] [killall[9]] [status]"
        sys.exit(-1)

    goal = sys.argv[1]

    r = redis.StrictRedis('knight-server.dyndns.org')

    if goal == 'sync':
        assert sys.argv[2] in syncgroups
        group = syncgroups[sys.argv[2]]
        if group.cde:
            updateCDE()
        #rsync to group
        sync(group)

    elif goal == 'syncall':
        #rsync to all in syncgroups
        updateCDE()
        for x in syncgroups.values():
            sync(x)

    elif goal == 'launch':
        assert sys.argv[2] in serverconfigs
        host = serverconfigs[sys.argv[2]]

        launchClient(host)

    elif goal == 'launchgroup':
        #for host in cesg_small:
        #for host in gsp_compute + 'toxic sequenceanalyze bana-desktop'.split():
        for host in cesg_small + 'camdi16 raptor hornet toxic sequenceanalyze bana-desktop'.split():
            cfg = serverconfigs[host]
            launchClient(cfg)

    elif goal == 'postdummy':
        from samcnet.utils import drawGraphs
        import networkx as nx
        import numpy as np
        from samcnet.generator import generateHourGlassGraph
        g = generateHourGlassGraph(4, 1)
        drawGraphs(g)
        base = dict(
            nodes = 15,
            samc_iters=1e4,
            numdata=200,
            priorweight=5,
            experiment_type='difference',
            gen_method = 'noisylogic',
            graph = np.array(nx.to_numpy_matrix(g),dtype=np.int32).tostring(),
            seed = 12341234,
            note = 'Fixed graph, small priorweight.',
            numtemplate=15)
        postJob(base, samples=10)

    elif goal == 'post':
        import networkx as nx
        import numpy as np
        N = 5
        g = getGraph(N)
        base = dict(
            nodes = N,
            samc_iters = 1e6,
            numdata = 0,
            priorweight = 4,
            experiment_type = 'single',
            gen_method = 'dirichlet',
            graph = np.array(nx.to_numpy_matrix(g),dtype=np.int32).tostring(),
            seed = 12341234,
            noise = 0.05,
            note = 'Cohesion: no data',
            stepscale = 100000,
            burn = 200000,
            truncate = 3,
            numtemplate=10)
        postJob(base, samples=10, single=True)

    elif goal == 'postsweep':
        import networkx as nx
        import numpy as np
        N = 8
        g = getGraph(N)
        base = dict(
            nodes = N,
            samc_iters=1e6,
            numdata='sweep',
            priorweight=4,
            experiment_type='single',
            gen_method = 'dirichlet',
            graph = np.array(nx.to_numpy_matrix(g),dtype=np.int32).tostring(),
            seed = 12341234,
            noise = 0.05,
            note = '1a with template',
            burn = 10000,
            stepscale = 100000,
            truncate = 3,
            numtemplate=10)
        postSweep(base, 10, 'numdata', [0,10,20,30,40,50,80])
        base['numtemplate'] = 0
        base['note'] = '1b without temp'
        postSweep(base, 10, 'numdata', [0,10,20,30,40,50,80])

    elif goal == 'cleanjobs':
        joblist = r.hgetall('desired-samples')
        for h,des in joblist.iteritems():
            if not r.exists(h) or r.llen(h) < 45:
                print 'Deleting hash: %s' % h
                r.delete(h)
                r.hdel('desired-samples', h)

    elif goal == 'killall':
        kill('all')

    elif goal == 'killall9':
        for host in gsp_compute + cesg_small + 'kubera raptor toxic sequenceanalyze'.split():
            cfg = serverconfigs[host]
            time.sleep(0.2)
            manualKill(cfg)
        r.delete('clients-hb')

    elif goal == 'kill':
        assert sys.argv[2] in serverconfigs
        kill(sys.argv[2])

    elif goal == 'status':
        if len(sys.argv) == 3:
            verbose=True
        else:
            verbose=False
        print("The job list is currently:")
        joblist = r.hgetall('desired-samples')
        for i,x in joblist.iteritems():
            #print '\t%s\t%s' % (r.hget('configs',i),x)
            if verbose or int(x) > 0:
                print '\t%s: %3s' % (i[:8],x)

        print 'Current sample counts:'
        for x in joblist.keys():
            count = r.llen(x)
            if verbose or count > 0:
                print '\t%s: %3d' % (x[:8],count)

        clients = r.zrevrange('clients-hb', 0, -1)
        num = len(clients)
        if num == 0:
            print('There are currently no clients alive.')
        else:
            print("The %d clients alive are:" % num)
            curr_time = r.time()
            cores = 0
            for x in clients:
                print '\t%s with hb %3.1f seconds ago' \
                        % (x, curr_time[0] + (curr_time[1]*1e-6) - int(r.zscore('clients-hb',x)))
                cores += serverconfigs[x].cores

            print("Total online cores: %d" % cores)


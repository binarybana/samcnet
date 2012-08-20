#!/usr/bin/env python
import os, sys, shlex, time, sha
import subprocess as sb
import redis
import simplejson as js
from samcnet.server_configs import serverconfigs, syncgroups, cesg_small, cesg_large, gsp_compute

LocalRoot = '/home/bana/GSP/research/samc/code'

def launchClient(host):
    cores = host.cores
    if host.cde:
        spec = ('ssh {0.hostname} cd {0.root}/cde-package/cde-root/' \
                + 'home/bana/GSP/research/samc/code; ').format(host) \
                + ('nohup {0.python} samcnet/driver.py {1} >/dev/null 2>&1 &'.format(host,cores))
    else:
        spec = ('ssh {0.hostname} cd {0.root};' + \
                'LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH nohup {0.python} '+ \
                'samcnet/driver.py {1} >/dev/null 2>&1 &').format(host,cores)

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
        print ("Beginning rsync to %s... " % group.hostname)
        p = sb.Popen('rsync -acz {0}/cde-package {1.hostname}:{1.dir}'.format(LocalRoot, group).split())
    else:
        print "Beginning rsync... "
        p = sb.Popen('rsync -acz --exclude=*cde* --exclude=build {0}/ {1.hostname}:{1.dir}/'.format(LocalRoot, group).split())
        print ' Done.'
        p.wait()
        print 'Beginning remote rebuild...'
        p = sb.Popen(shlex.split('ssh {0.hostname} "cd {0.dir}; ./waf distclean; . cfg; ./waf"'.format(group)))
    p.wait()
    print ' Done.'

def updateCDE():
    print "Updating CDE package..." 
    os.environ['LD_LIBRARY_PATH']='./build:../build'
    os.chdir(LocalRoot)
    p = sb.Popen('/home/bana/bin/cde python {0}/samcnet/driver.py rebuild'.format(LocalRoot).split())
    p.wait()
    p = sb.Popen('rsync -a samcnet cde-package/cde-root/home/bana/GSP/research/samc/code/'.format(LocalRoot).split())
    p.wait()
    p = sb.Popen('rsync -a build cde-package/cde-root/home/bana/GSP/research/samc/code/'.format(LocalRoot).split())
    p.wait()
    print " Done."

def postJob(job, samples):
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
    tot = r.hincrby('desired-samples', h, samples)
    print("Added %d samples for a total of %d samples remaining." % (samples,tot))
    print("Pushed job:" )
    for k,v in job.iteritems():
        if k == 'graph':
            print '\t'+k+':\t'+str(hash(v))
        else:
            print '\t'+k+':\t'+str(v)

def postSweep(base, iters, param, values):
    """ 
    Take the <base> config, and get <iters> samples across the <values> in <param>.
    Also save the base config in 'sweep-configs' in Redis.
    """
    assert param in base
    base[param] = 'sweep'
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
        for host in gsp_compute + cesg_small + 'raptor blackbird compute-0-5 camdi16 toxic sequenceanalyze bana-desktop'.split():
            cfg = serverconfigs[host]
            launchClient(cfg)

    elif goal == 'postdummy':
        from samcnet.utils import drawGraphs
        import networkx as nx
        import numpy as np
        from samcnet.generator import generateHourGlassGraph
        g = generateHourGlassGraph(15, 2)
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
        #test = dict(
            #nodes = 5,
            #samc_iters=1e4,
            #numdata=50,
            #priorweight=10,
            #numtemplate=5)
        postJob(base, samples=1)

    elif goal == 'post':
        test = dict(
            nodes = 20,
            samc_iters=3e5,
            numdata=10,
            priorweight=500,
            numtemplate=15)
        postJob(test, samples=20)

    elif goal == 'postsweep':
        from samcnet.utils import drawGraphs
        import networkx as nx
        import numpy as np
        from samcnet.generator import generateHourGlassGraph
        cont = 'n'
        while cont != 'y':
            g = generateHourGlassGraph(40, 3)
            drawGraphs(g)
            cont = raw_input('Is this graph okay? (y/n): ')
        base = dict(
            nodes = 40,
            samc_iters=5e5,
            numdata='sweep',
            priorweight=40,
            experiment_type='difference',
            gen_method = 'noisylogic',
            graph = np.array(nx.to_numpy_matrix(g),dtype=np.int32).tostring(),
            seed = 12341234,
            noise = 0.05,
            note = 'latenight - one more swing at sweeping across numdata, this time with a fixed graph.',
            numtemplate=30)
        postSweep(base, 200, 'numtemplate', [20,50,100,200,500])

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

            print("Cores active: %d" % cores)
        print("The job list is currently:")
        joblist = r.hgetall('desired-samples')
        for i,x in joblist.iteritems():
            #print '\t%s\t%s' % (r.hget('configs',i),x)
            print '\t%s: %s' % (i,x)

        print 'Current sample counts:'
        for x in joblist.keys():
            print '\t%s: %3d' % (x,r.llen(x))


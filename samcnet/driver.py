import sys, os
import redis
import sha
import atexit
import subprocess as sb
from time import time, sleep
import traceback
try:
    import simplejson as js
except:
    import json as js
import logging
import logging.handlers

h = logging.handlers.SysLogHandler(('knight-server.dyndns.org',10514))
h.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s: samc %(levelname)s %(message)s')
h.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(h)
logger.setLevel(logging.DEBUG)

def log_uncaught_exceptions(ex_cls, ex, tb):
    logger.critical(''.join(traceback.format_tb(tb)))
    logger.critical('{0}: {1}'.format(ex_cls, ex))

sys.excepthook = log_uncaught_exceptions

logger.info('Connecting to db.')
r = redis.StrictRedis('knight-server.dyndns.org')

def getHost():
    return os.uname()[1].split('.')[0]

def recordDeath():
    r.zrem('clients-hb', getHost())

def free(x):
    return x == None or x.poll() != None

def spawn(job, workhash):
    env = os.environ
    env['SAMC_JOB'] = job
    env['WORKHASH'] = workhash
    spec = 'python samcnet/experiment.py'
    return sb.Popen(spec.split(), env=env)

def kill(spawn):
    if spawn == None:
        return
    else: 
        spawn.kill()

atexit.register(recordDeath)

if len(sys.argv) != 2:
        print 'Usage: ./driver.py <number of children>'
        sys.exit()

if sys.argv[1] == 'rebuild':
    logger.info('Beginning dummy run for CDE rebuild')
    capacity = 1
    children = [None] * capacity
    cmd = r.get('die')
    if getHost() == 'dummy':
        x = 2
    test = dict(
            nodes = 5, 
            samc_iters=1e4, 
            numdata=50, 
            priorweight=10, 
            burn=0,
            data_method='dirichlet',
            numtemplate=5)
    test = js.dumps(test)
    x = spawn(test, sha.sha(test).hexdigest())
    x.wait()
    sys.exit()

else:
    capacity = int(sys.argv[1])
    children = [None] * capacity
    if r.zscore('clients-hb', getHost()):
        logger.warning('It appears there already exists a HB client on '+\
                'this host, shutting down')
        sys.exit()

while True:
    r.zadd('clients-hb', r.time()[0], getHost())
    cmd = r.get('die')
    if cmd == 'all' or cmd == getHost():
        logger.info("Received die command, shutting down.")
        for x in children:
            kill(x)
        r.zrem('clients-hb', getHost())
        break

    freelist = filter(free, children)
    if len(freelist) > 0:
        logger.info('Found %d free children', len(freelist))
    for x in freelist:
        children.remove(x)
    del freelist
    freenum = capacity - len(children)
    workhash = None
    if freenum > 0:
        with r.pipeline(transaction=True) as pipe:
            while True:
                try:
                    workhash = None
                    pipe.watch('desired-samples')
                    queue = pipe.hgetall('desired-samples')
                    for h,num in queue.iteritems():
                        if int(num) > 0:
                            logger.info("Found %s samples left on hash %s" % (num, h))
                            workhash = h
                            break
                    if workhash != None:
                        # We found some work!
                        grab = freenum if freenum < int(num) else int(num)
                        logger.debug('Freenum: %d, desirednum: %d, grab: %d', freenum, int(num), grab)
                        pipe.multi()
                        pipe.hincrby('desired-samples', workhash, -1*grab)
                        pipe.execute()
                    break
                except redis.WatchError:
                    continue
            pipe.unwatch()


    if workhash == None:
        logger.debug('sleeping for 2 seconds...')
        sleep(2)
        continue
    else:
        job = r.hget('configs', workhash)
        logger.info('Spawning %d new children', grab)
        newchildren = [spawn(job, workhash) for x in range(grab)]
        children.extend(newchildren)
        

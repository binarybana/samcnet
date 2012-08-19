import sys, os
import redis
import sha
import atexit
import subprocess as sb
import logging
import logging.handlers
try:
    import simplejson as js
except:
    import json as js

h = logging.handlers.SysLogHandler(('knight-server.dyndns.org',10514))
h.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s: samc %(levelname)s %(message)s')
h.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(h)
logger.setLevel(logging.DEBUG)


from time import time, sleep

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
    #return sb.Popen(spec.split(), env=env, stdout=null)

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
    null = open('/dev/null','a')
    cmd = r.get('die')
    if getHost() == 'dummy':
        x = 2
    test = dict(
            nodes = 5, 
            samc_iters=1e4, 
            numdata=50, 
            priorweight=10, 
            burn=0,
            numtemplate=5)
    test = js.dumps(test)
    x = spawn(test, sha.sha(test).hexdigest())
    x.wait()
    sys.exit()

else:
    capacity = int(sys.argv[1])
    children = [None] * capacity
    null = open('/dev/null','a')

logger.info('Connecting to db.')
while True:
    r.zadd('clients-hb', r.time()[0], getHost())
    cmd = r.get('die')
    if cmd == 'all' or cmd == getHost():
        logging.info("Received die command, shutting down.")
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
            queue = r.hgetall('desired-samples')
            for h,num in queue.iteritems():
                if int(num) > 0:
                    logging.info("Found %s samples left on hash %s" % (num, h))
                    workhash = h
                    break
            if workhash != None:
                # We found some work!
                #grab = freenum if freenum < num else num
                if freenum < num:
                    grab = freenum
                else:
                    grab = num
                r.hincrby('desired-samples', workhash, -1*grab)

    if workhash == None:
        logging.debug('sleeping for 2 seconds...')
        sleep(2)
        continue
    else:
        job = r.hget('configs', workhash)
        logging.info('Spawning %d new children', grab)
        newchildren = [spawn(job, workhash) for x in range(grab)]
        children.extend(newchildren)
        





import sys
import os
import sha
import atexit
import traceback
import logging
import logging.handlers
import subprocess as sb
from time import time, sleep
import uuid
try:
    import simplejson as js
except:
    import json as js

import redis
from utils import getHost

try:
    syslog_server = os.environ['SYSLOG']
    redis_server = os.environ['REDIS']
except:
    print "ERROR: Need SYSLOG and REDIS environment variables defined."
    sys.exit(1)

h = logging.handlers.SysLogHandler((syslog_server,10514))
h.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s: samc %(levelname)s %(message)s')
h.setFormatter(formatter)
logger = logging.getLogger(getHost() + ' driver')
#logger = logging.getLogger()
logger.addHandler(h)
logger.setLevel(logging.DEBUG)

def log_uncaught_exceptions(ex_cls, ex, tb):
    logger.critical(''.join(traceback.format_tb(tb)))
    logger.critical('{0}: {1}'.format(ex_cls, ex))

sys.excepthook = log_uncaught_exceptions

r = None

def recordDeath():
    if r is not None:
        r.zrem('clients-hb', getHost())

def free(x):
    return x == None or x.poll() != None

def spawn(job, workhash):
    env = os.environ
    env['SAMC_JOB'] = job
    env['WORKHASH'] = workhash
    env['LD_LIBRARY_PATH'] = '/share/apps/lib:.:lib:build'
    spec = 'python -m samcnet.experiment'
    #fid = open('/tmp/log','w')
    return sb.Popen(spec.split(), env=env)
    #return sb.Popen(spec.split(), env=env, stdout=fid, stderr=fid)

def kill(spawn):
    if spawn == None:
        return
    else: 
        spawn.kill()


if __name__ == '__main__':
    if len(sys.argv) != 2:
            print 'Usage: ./driver.py <number of children>'
            sys.exit()
    elif sys.argv[1] == 'rebuild':
        logger.info('Beginning dummy run for CDE rebuild')
        test = dict(
                nodes = 5, 
                samc_iters=10, 
                numdata=5, 
                priorweight=1, 
                burn=0,
                data_method='dirichlet',
                numtemplate=5)
        test = js.dumps(test)
        x = spawn(test, sha.sha(test).hexdigest())
        x.wait()
        sys.exit()
    else:
        logger.info('Connecting to db.')
        r = redis.StrictRedis(redis_server)
        atexit.register(recordDeath)
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
                if x is not None and x.returncode != 0:
                    logger.warning("Child returned error return code %d", x.returncode)
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
                #logger.debug('sleeping for 2 seconds...')
                sleep(2)
                continue
            else:
                job = r.hget('configs', workhash)
                logger.info('Spawning %d new children', grab)
                newchildren = [spawn(job, workhash) for x in range(grab)]
                children.extend(newchildren)
                

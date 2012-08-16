#!/usr/bin/env python
import os, sys, shlex, time, sha
import subprocess as sb
import redis
import random
from collections import namedtuple, Counter

ServerConfig = namedtuple('ServerConfig', 'hostname root python cde cores') 
SyncGroup = namedtuple('SyncGroup', 'hostname dir cde')

LocalRoot = '/home/bana/GSP/research/samc/code'

def launchClient(host):
  cores = host.cores
  if host.cde:
    spec = ('ssh {0.hostname} cd {0.root}/cde-package/cde-root/' \
        + 'home/bana/GSP/research/samc/code; ').format(host) \
        + ('{0.python} py/driver.py {1} &'.format(host,cores))
  else:
    spec = ('ssh {0.hostname} cd {0.root};'.format(host) \
        + ('LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH {0.python} py/driver.py {1} &'.format(host,cores)))

  print "Connecting to %s." % host.hostname
  sb.Popen(shlex.split(spec), 
      bufsize=-1,
      stdout=open('/tmp/samc-{0.hostname}-{1}.log'.format(host, random.randint(0,1e9)) ,'w'))
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
  p = sb.Popen('/home/bana/bin/cde python {0}/py/driver.py rebuild'.format(LocalRoot).split())
  p.wait()
  p = sb.Popen('rsync -a py cde-package/cde-root/home/bana/GSP/research/samc/code/'.format(LocalRoot).split())
  p.wait()
  p = sb.Popen('rsync -a build cde-package/cde-root/home/bana/GSP/research/samc/code/'.format(LocalRoot).split())
  p.wait()
  print " Done."

def postJob(job, samples):
  num = r.zcard('clients-hb')
  if num == 0:
    print "No clients online, aborting."
  else:
    h = sha.sha(job).hexdigest()
    r.hsetnx('configs', h, job)
    tot = r.hincrby('desired-samples', h, samples)
    print("Added %d samples for a total of %d samples remaining." % (samples,tot))
    print("Pushed to %d clients. Job:" % num)
    for x in job.split():
      print '\t'+x

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
      clients = r.sort('clients-hb')
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

if len(sys.argv) < 2:
  print "Usage: python exec.py [sync <groupname>] [syncall]"+\
      " [postdummy] [kill <hostname>] [killall[9]] [status]"
  sys.exit(-1)

goal = sys.argv[1]

r = redis.StrictRedis('knight-server.dyndns.org')

syncgroups = {'cesg': SyncGroup('raptor', '/home/bana', True),
    'gsp': SyncGroup('kubera', '/home/jason/samc/samcsynthetic-0.2', False),
    'camdi': SyncGroup('camdi', '/home/bana/GSP/research/samc/code', False),
    'toxic': SyncGroup('toxic','/home/jason', True),
    'sequencer': SyncGroup('sequencer','/home/jason', True)}

serverconfigs = {}

cesg_small = 'mustang spitfire kingcobra zero'.split()
for x in cesg_small:
  serverconfigs[x] = ServerConfig(x, 
      '/home/bana',  
      '/home/bana/cde-package/cde-exec python', 
      True,
      6)
cesg_large = 'raptor blackbird hornet'.split()
for x in cesg_large:
  serverconfigs[x] = ServerConfig(x, 
      '/home/bana',  
      '/home/bana/cde-package/cde-exec python', 
      True,
      20)

gsp_compute = ['compute-0-%d'%x for x in range(1,5)] + ['kubera']
for x in  gsp_compute + ['kubera']:
  serverconfigs[x] = ServerConfig(x, 
      '/home/jason/samc/samcsynthetic-0.2', 
      '/share/apps/bin/python2.7', 
      False,
      8)

serverconfigs['toxic'] = ServerConfig('toxic', 
      '/home/jason', 
      '/home/jason/cde-package/cde-exec python', 
      True,
      5)
serverconfigs['sequencer'] = ServerConfig('sequencer', 
      '/home/jason', 
      '/home/jason/cde-package/cde-exec python', 
      True,
      10)
serverconfigs['bana-desktop'] = ServerConfig('bana-desktop', 
      '/home/bana/GSP/research/samc/code', 
      'python', 
      False,
      2)
serverconfigs['camdi'] = ServerConfig('camdi', 
      '/home/bana/GSP/research/samc/code', 
      'python', 
      False,
      4)

if goal == 'sync':
  assert sys.argv[2] in syncgroups
  group = syncgroups[sys.argv[2]]
  if group.cde:
    updateCDE()
  #rsync to group
  sync(group)

elif goal == 'syncall':
  #rsync to all in syncgroups
  for x in syncgroups.values():
    sync(x)

elif goal == 'launch':
  assert sys.argv[2] in serverconfigs
  host = serverconfigs[sys.argv[2]]

  launchClient(host)

elif goal == 'launchgroup':
  #for host in cesg_small:
  #for host in gsp_compute + 'toxic sequencer bana-desktop'.split():
  for host in gsp_compute + cesg_small + 'raptor toxic sequencer bana-desktop'.split():
    cfg = serverconfigs[host]
    time.sleep(0.2)
    launchClient(cfg)

elif goal == 'postdummy':
  test = """
[General]
nodes = 5
samc-iters=1e4
numdata=50
priorweight=10
numtemplate=5"""
  postJob(test, samples=24)

elif goal == 'post':
  test = """
[General]
nodes = 20
samc-iters=3e5
numdata=10
priorweight=500
numtemplate=15"""
  postJob(test, samples=200)

elif goal == 'killall':
  kill('all')

elif goal == 'killall9':
  for host in gsp_compute + cesg_small + 'kubera raptor toxic sequencer'.split():
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
    for x in clients:
      print '\t%s with hb %3.1f seconds ago' \
          % (x, curr_time[0] + (curr_time[1]*1e-6) - int(r.zscore('clients-hb',x)))
  print("The job list is currently:")
  joblist = r.hgetall('desired-samples')
  for i,x in joblist.iteritems():
    #print '\t%s\t%s' % (r.hget('configs',i),x)
    print '\t%s: %s' % (i,x)

  print 'Current sample counts:'
  for x in joblist.keys():
    print '\t%s: %3d' % (x,r.llen(x))


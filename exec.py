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
        + ('{0.python} py/driver.py &'.format(host))*cores
  else:
    spec = ('ssh {0.hostname} cd {0.root};'.format(host) \
        + ('LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH {0.python} py/driver.py &'.format(host))*cores)

  print "Connecting to %s." % host.hostname
  sb.Popen(shlex.split(spec), 
      bufsize=-1,
      stdout=open('/tmp/samc-{0.hostname}-{1}.log'.format(host, random.randint(0,1e9)) ,'w'))
  return 

def sync(group):
  if group.cde:
    print "Beginning rsync to %s... " % group
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
  p = sb.Popen('/home/bana/bin/cde python {0}/py/driver.py'.format(LocalRoot).split())
  time.sleep(1)
  p2 = sb.Popen('./exec.py postdummy'.split())
  p2.wait()
  p2 = sb.Popen('./exec.py kill {0}'.format(os.uname()[1].split('.')[0]).split())
  p2.wait()
  p.wait()
  print " Done."

def postJob(job, samples):
  numcl = r.llen('clients-alive')
  if numcl == 0:
    print "No clients online, aborting."
  else:
    h = sha.sha(job).hexdigest()
    tot = r.hincrby('desired-samples', h, samples)
    r.hsetnx('configs', h, job)
    num = r.llen('clients-alive')
    print("Added %d samples for a total of %d samples remaining." % (samples,tot))
    print("Pushed to %d clients. Job:" % num)
    for x in job.split():
      print '\t'+x

def kill(target):
  assert not r.exists('die')
  print "Sending out kill command to %s." % target
  r.set('die', target)

  def countTarget(target):
    if target == 'all':
      return r.llen('clients-alive')
    else:
      return len([x for x in r.lrange('clients-alive', 0, -1) if x == target])

  try:
    tot = num = countTarget(target)
    print ('Waiting for %d clients to die...' % num)

    while num > 0:
      time.sleep(1)
      num = countTarget(target)

    print("%d clients killed." % tot)
  finally:
    r.delete('die')

if len(sys.argv) < 2:
  print "Usage: python exec.py [sync <groupname>] [syncall]"+\
      " [postdummy] [kill <hostname>] [killall] [status]"
  sys.exit(-1)

goal = sys.argv[1]

r = redis.StrictRedis('knight-server.dyndns.org')

syncgroups = {'cesg': SyncGroup('hornet', '/home/bana', True),
    'gsp': SyncGroup('kubera', '/home/jason/samc/samcsynthetic-0.2', False),
    'toxic': SyncGroup('toxic','/home/jason', True),
    'sequencer': SyncGroup('sequencer','/home/jason', True)}

serverconfigs = {}

cesg_small = 'mustang spitfire kingcobra zero'.split()
for x in cesg_small:
  serverconfigs[x] = ServerConfig(x, 
      '/home/bana',  
      '/home/bana/cde-package/cde-exec python', 
      True,
      4)
cesg_large = 'raptor blackbird hornet'.split()
for x in cesg_large:
  serverconfigs[x] = ServerConfig(x, 
      '/home/bana',  
      '/home/bana/cde-package/cde-exec python', 
      True,
      20)

gsp_compute = ['compute-0-%d'%x for x in range(1,4)]
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
      12)
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
  for host in cesg_small:
  #for host in gsp_compute + 'toxic sequencer bana-desktop'.split():
  #for host in gsp_compute + cesg_small + 'raptor toxic sequencer bana-desktop'.split():
    cfg = serverconfigs[host]
    time.sleep(0.2)
    launchClient(cfg)

elif goal == 'postdummy':
  test = """
[General]
nodes = 15
samc-iters=5e5
numdata=500
priorweight=10
numtemplate=5"""
  postJob(test, samples=20)

elif goal == 'killall':
  kill('all')

elif goal == 'kill':
  assert sys.argv[2] in serverconfigs
  kill(sys.argv[2])

elif goal == 'status':
  r.sort('clients-alive')
  clients = r.lrange('clients-alive', 0, -1)
  num = len(clients)
  if num == 0:
    print('There are currently no clients alive.')
  else:
    print("The %d clients alive are:" % num)
    cores = Counter(clients)
    clients = list(set(clients))
    clients.sort()
    for x in clients:
      print '\t%3d cores on %s' %(cores[x],x)
  print("The job list is currently:")
  joblist = r.hgetall('desired-samples')
  for i,x in joblist.iteritems():
    #print '\t%s\t%s' % (r.hget('configs',i),x)
    print '\t%s: %s' % (i,x)

  print 'Current sample counts:'
  for x in joblist.keys():
    print '\t%s: %3d' % (x,r.llen(x))


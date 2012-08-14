#!/usr/bin/env python
import execnet as ex
import os, sys, subprocess, shlex
from collections import namedtuple

ServerConfig = namedtuple('ServerConfig', 'hostname root python cde cores') 
SyncGroup = namedtuple('SyncGroup', 'hostname dir cde')

LocalRoot = '/home/bana/GSP/research/samc/code'

def makeGateway(host):
  if host.cde:
    spec = ('ssh={0.hostname}//chdir={0.root}/cde-package/cde-root/' + \
        'home/bana/GSP/research/samc/code/py//python={0.python}').format(host)
  else:
    spec = ('ssh={0.hostname}//chdir={0.root}/py//python={0.python}').format(host)

  print "Connecting to: "
  for x in spec.split('//'):
    print '\t'+x
  gw = ex.makegateway(spec)
  return gw

def sync(group):
  if group.cde:
    updateCDE()
    p = subprocess.Popen('rsync -aczP --delete {0}/cde-package {1.hostname}:{1.dir}'.format(LocalRoot, group).split())
  else:
    p = subprocess.Popen('rsync -aczP --delete --exclude=*cde* --exclude=build {0}/ {1.hostname}:{1.dir}/'.format(LocalRoot, group).split())
    p.wait()
    p = subprocess.Popen(shlex.split('ssh {0.hostname} "cd {0.dir}; ./waf distclean; . cfg; ./waf"'.format(group)))
  return p.wait()

def updateCDE():
  os.environ['LD_LIBRARY_PATH']='./build:../build'
  os.chdir(LocalRoot)
  p = subprocess.Popen('/home/bana/bin/cde python {0}/py/driver.py'.format(LocalRoot).split())
  p.wait()

if len(sys.argv) < 2:
  print "Usage: python exec.py [synchost <hostname>] [syncall] [testrun <hostname>] [run]"
  sys.exit(-1)

goal = sys.argv[1]

syncgroups = {'cesg': SyncGroup('hornet', '/home/bana', True),
    'gsp': SyncGroup('kubera', '/home/jason/samc/samcsynthetic-0.2', False),
    'toxic': SyncGroup('toxic','/home/jason', True),
    'sequencer': SyncGroup('sequencer','/home/jason', True)}

serverconfigs = {}

for x in 'mustang spitfire kingcobra zero'.split():
  serverconfigs[x] = ServerConfig(x, 
      '/home/bana',  
      '/home/bana/cde-package/cde-exec python', 
      True,
      8)

for x in 'raptor blackbird hornet'.split():
  serverconfigs[x] = ServerConfig(x, 
      '/home/bana',  
      '/home/bana/cde-package/cde-exec python', 
      True,
      47)
for x in ['compute-0-1', 'compute-0-2', 'compute-0-3', 'kubera']:
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
      12)

if goal == 'syncgroup':
  assert sys.argv[2] in syncgroups
  group = syncgroups[sys.argv[2]]
  #rsync to group
  sync(group)

elif goal == 'syncall':
  #rsync to all in syncgroups
  for x in syncgroups.values():
    sync(x)

elif goal == 'testrun':
  assert sys.argv[2] in serverconfigs
  host = serverconfigs[sys.argv[2]]
  sys.path.append('../code/py')
  import driver

  mch = ex.MultiChannel([makeGateway(host).remote_exec(driver) for x in range(host.cores)])

  #mch = ex.MultiChannel([makeGateway(host).remote_exec("channel.send(channel.receive()+1)") for x in range(host.cores)])
  #mch.send_each(1)

  #ch = gw.remote_exec(driver)
  #ch = gw.remote_exec("channel.send(channel.receive()+1)")

  print mch.receive_each()

elif goal == 'run':
  pass
  #Run all

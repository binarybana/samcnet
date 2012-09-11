from collections import namedtuple

ServerConfig = namedtuple('ServerConfig', 'hostname root python cde cores') 
SyncGroup = namedtuple('SyncGroup', 'hostname dir cde')

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
      7)

cesg_large = 'raptor blackbird hornet'.split()
cores = [20,25,20]
for c,x in zip(cores, cesg_large):
  serverconfigs[x] = ServerConfig(x, 
      '/home/bana',  
      '/home/bana/cde-package/cde-exec python', 
      True,
      c)

gsp_compute = ['compute-0-%d'%x for x in range(1,4)] + ['kubera']
gsp_compute_all = ['compute-0-%d'%x for x in range(1,8)] + ['kubera']
for x in gsp_compute_all:
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
serverconfigs['sequenceanalyze'] = ServerConfig('sequenceanalyze', 
      '/home/jason', 
      '/home/jason/cde-package/cde-exec python', 
      True,
      12)
serverconfigs['bana-desktop'] = ServerConfig('bana-desktop', 
      '/home/bana/GSP/research/samc/code', 
      'python', 
      False,
      3)
serverconfigs['camdi16'] = ServerConfig('camdi16', 
      '/home/bana/GSP/research/samc/code', 
      'python', 
      False,
      4)

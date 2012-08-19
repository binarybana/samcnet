from collections import namedtuple, Counter

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

from collections import namedtuple

### Definitions, don't change these ####
ServerConfig = namedtuple('ServerConfig', 'hostname root python cde cores') 
SyncGroup = namedtuple('SyncGroup', 'hostname dir cde')
########################################

cfg = {}

### Define local redis server, local root, sync groups and servers ###
cfg['redis_server'] = "camdi16.tamu.edu"
cfg['syslog_server'] = "camdi16.tamu.edu"
cfg['local_root'] = '/home/bana/GSP/research/samc/code'

#cfg['sync_groups'] = {'cesg': SyncGroup('raptor', '/home/bana', True),
    #'gsp': SyncGroup('kubera', '/home/jason/samc/samcsynthetic-0.2', False),
    #'camdi16': SyncGroup('camdi16', '/home/bana/GSP/research/samc/code', False),
    #'toxic': SyncGroup('toxic','/home/jason', True),
    #'sequencer': SyncGroup('sequencer','/home/jason', True)}
cfg['sync_groups'] = {'wsgi' : SyncGroup('wsgi', '/home/binarybana', True)}

### Define Servers ###
cfg['server_configs'] = {}

cfg['launch_group'] = '' # for now....

cesg_small = 'mustang spitfire kingcobra zero'.split()
for x in cesg_small:
  cfg['server_configs'][x] = ServerConfig(x, 
      '/home/bana',  
      '/home/bana/cde-package/cde-exec python', 
      True,
      7)

cesg_large = 'raptor hornet blackbird'.split()# blackbird
cores = [40,40,40]
for c,x in zip(cores, cesg_large):
  cfg['server_configs'][x] = ServerConfig(x, 
      '/home/bana',  
      '/home/bana/cde-package/cde-exec python', 
      True,
      c)

gsp_compute = ['compute-0-%d'%x for x in range(4,8)] #+ ['kubera']
gsp_compute_all = ['compute-0-%d'%x for x in range(1,8)] + ['kubera']
for x in gsp_compute_all:
  cfg['server_configs'][x] = ServerConfig(x, 
      '/home/jason/samc/samcsynthetic-0.2', 
      '/share/apps/bin/python2.7', 
      False,
      8)

cfg['server_configs']['toxic'] = ServerConfig('toxic', 
      '/home/jason', 
      '/home/jason/cde-package/cde-exec python', 
      True,
      12)
cfg['server_configs']['sequenceanalyze'] = ServerConfig('sequenceanalyze', 
      '/home/jason', 
      '/home/jason/cde-package/cde-exec python', 
      True,
      12)
cfg['server_configs']['bana-desktop'] = ServerConfig('bana-desktop', 
      '/home/bana/GSP/research/samc/code', 
      'python', 
      False,
      3)
cfg['server_configs']['camdi16'] = ServerConfig('camdi16', 
      '/home/bana/GSP/research/samc/code', 
      'python', 
      False,
      4)

import sys, os

if len(sys.argv) == 2:
    os.environ['SAMC_JOB_RUNS'] = '100'
elif len(sys.argv) == 3:
    os.environ['SAMC_JOB_RUNS'] = sys.argv[2]
else:
    print "Usage: ./metadriver <.cfg file>"
    sys.exit(-1)

os.environ['SAMC_JOB_CFG'] = sys.argv[1]

os.popen('qsub mpirunner.pbs')

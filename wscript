#! /usr/bin/env python
# encoding: utf-8
#
APPNAME = 'libsamc'
VERSION = '0.1'

top = '.'
out = 'build'

def options(opt):
    opt.load('compiler_c python cython')
    opt.add_option('-d', '--debug', action='store_true', default=False, help='Debug flag.')
    opt.add_option('-p', '--prof', action='store_true', default=False, help='Profiling flag.')

def configure(conf):
    #conf.env.CFLAGS = "-O2 -fPIC -shared".split()
    #conf.env.CFLAGS = "-O2 -fPIC -shared".split()
    conf.load('compiler_c python cython')
    conf.check_python_headers()
    conf.check(compiler='cc', lib='Judy', uselib_store='JUDY')
    conf.check(compiler='cc', lib='m', uselib_store='MATH')
    #conf.env.debug = conf.options.debug
    #conf.check_cc(header_name="Judy.h")
    #conf.check_cc(header_name="math.h")
    #features='cxx', lib=['m','Judy'], cflags='-Wall -O2'.split())

def build(bld):
    #bld.shlib(source=bld.path.ant_glob('./src/*.c'), target='samc')
    #if(bld.env.debug):

    CFLAGS = ['-Wall','-std=c99']
    LDFLAGS = []
    if bld.options.debug:
        print('Beginning debug build')
        CFLAGS += ['-g']
        LDFLAGS += ['-g']
    if bld.options.prof:
        print('Adding profiling flag build')
        CFLAGS += ['-pg']
        LDFLAGS += ['-pg']
    if not bld.options.prof and not bld.options.debug: 
        CFLAGS += ['-O2']

    libs = 'JUDY MATH'.split()
    #bld.program(source=bld.path.ant_glob('src/*.c'), 
                #target='samcapp', 
                #cflags=CFLAGS,
                #linkflags=LDFLAGS,
                #use=libs)

    bld.shlib(source=bld.path.ant_glob('src/*.c'), 
                target='samc', 
                cflags=CFLAGS,
                linkflags=LDFLAGS,
                use=libs)

    bld(features='c cshlib pyext',
        source=['py/snet.pyx'],
        includes=['src'],
        use='samc',
        libpath=['.','./build'],
        target='snet')

def dist(ctx):
        ctx.excl      = '**/*.zip **/*.bz2 **/.waf-1* **/*~ **/*.pyc **/*.swp **/.lock-w*' 
        #ctx.files     = ctx.path.ant_glob('**/wscript **/*.h **/*.cpp waf') 


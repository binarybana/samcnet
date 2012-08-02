#! /usr/bin/env python
# encoding: utf-8

APPNAME = 'samcsynthetic'
VERSION = '0.2'

top = '.'
out = 'build'

def options(opt):
    opt.load('compiler_c python cython')
    opt.add_option('-d', '--debug', action='store_true', default=False, help='Debug flag.')
    opt.add_option('-p', '--prof', action='store_true', default=False, help='Profiling flag.')

def configure(conf):
    conf.load('compiler_c python cython')
    conf.check_python_headers()
    conf.check(compiler='cc', lib='Judy', uselib_store='JUDY')
    conf.check(compiler='cc', lib='m', uselib_store='MATH')
    conf.check(compiler='cc', lib='profiler', uselib_store='PROFILER')

def build(bld):
    libs = 'JUDY MATH'.split()

    CFLAGS = ['-Wall','-std=c99']
    LDFLAGS = []
    CYTHONFLAGS = []
    if bld.options.debug:
        print('Beginning debug build')
        CFLAGS += ['-g','-DDEBUG']
        CYTHONFLAGS += ['--gdb']
        LDFLAGS += ['-g','-DDEBUG']
    if bld.options.prof:
        print('Adding profiling flag build')
        CFLAGS += ['-pg']
        LDFLAGS += ['-pg']
        libs += ['PROFILER']
    if not bld.options.prof and not bld.options.debug: 
        CFLAGS += ['-O2']

    bld.env.CYTHONFLAGS = CYTHONFLAGS

    bld.shlib(source = bld.path.ant_glob('src/*.c'), 
                target='cost', 
                cflags=CFLAGS,
                linkflags=LDFLAGS,
                use=libs)

    bld(features='c cshlib pyext',
        source=['py/samc.pyx'],
        includes=[],
        libpath=['.','./build'],
        target='samc')

    bld(features='c cshlib pyext',
        source=['py/bayesnet.pyx'],
        includes=['src'],
        use='cost',
        libpath=['.','./build'],
        target='bayesnet')

def dist(ctx):
        ctx.excl      = '**/*.zip **/*.bz2 **/.waf-1* **/*~ **/*.pyc **/*.swp **/.lock-w*' 
        #ctx.files     = ctx.path.ant_glob('**/wscript **/*.h **/*.cpp waf') 


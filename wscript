#! /usr/bin/env python
# encoding: utf-8

APPNAME = 'samcsynthetic'
VERSION = '0.4'

top = '.'
out = 'build'

def options(opt):
    opt.load('compiler_c compiler_cxx python cython')
    opt.add_option('-d', '--debug', action='store_true', default=False, help='Debug flag.')
    opt.add_option('-p', '--prof', action='store_true', default=False, help='Profiling flag.')

def configure(conf):
    conf.load('compiler_c compiler_cxx python cython')
    conf.check_python_headers()
    conf.check_python_module('numpy')
    conf.check_python_module('scipy')
    conf.check_python_module('networkx')
    conf.check_python_module('pandas')
    conf.env.append_value('LINKFLAGS', '-L%s/lib' % conf.path.abspath())
    conf.env.append_value('LINKFLAGS', '-L/share/apps/lib')
    conf.check(compiler='cc', lib='Judy', uselib_store='JUDY')
    conf.check(compiler='cc', lib='m', uselib_store='MATH')
    #conf.check(compiler='cc', lib='profiler', uselib_store='PROF')
    conf.check_cxx(lib='dai', uselib_store='DAI')
    conf.check_cxx(lib='gmp', uselib_store='GMP')
    conf.check_cxx(lib='gmpxx', uselib_store='GMPXX')

def build(bld):
    libs = 'JUDY MATH'.split()
    includes = ['-I/share/apps/include']

    CFLAGS = ['-Wall','-std=c99'] + includes
    CXXFLAGS = ['-fPIC'] + includes
    LDFLAGS = []
    CYTHONFLAGS = []
    if bld.options.debug:
        print('Beginning debug build')
        CFLAGS += ['-g','-DDEBUG']
        CXXFLAGS += ['-g','-DDEBUG']
        CYTHONFLAGS += ['--gdb']
        LDFLAGS += ['-g','-DDEBUG']
    if bld.options.prof:
        print('Adding profiling flag build')
        CFLAGS += ['-pg']
        CXXFLAGS += ['-pg']
        LDFLAGS += ['-pg']
        #libs += ['PROF']
    if not bld.options.prof and not bld.options.debug: 
        CFLAGS += ['-O2']
        CXXFLAGS += ['-O2']

    bld.env.CYTHONFLAGS = CYTHONFLAGS

    bld.env['PREFIX'] = '.'

    bld.shlib(source = bld.path.ant_glob('samcnet/netcost/*.c'), 
        target='cost', 
        cflags=CFLAGS,
        linkflags=LDFLAGS,
        use=libs)

    bld(features='c cshlib pyext',
        source=['samcnet/samc.pyx'],
        includes=[],
        libpath=['.','./build'],
        cflags=CFLAGS,
        ldflags=LDFLAGS,
        target='samc')

    bld(features='c cshlib pyext',
        source=['samcnet/bayesnet.pyx'],
        includes=['samcnet/netcost'],
        use='cost',
        libpath=['.','./build'],
        cflags=CFLAGS,
        ldflags=LDFLAGS,
        target='bayesnet')

    bld(features='c cshlib cxx pyext',
        source=['samcnet/bayesnetcpd.pyx'],
        includes=['include'],
        #libpath=['.','./build'],
        cxxflags=CXXFLAGS,
        ldflags=LDFLAGS,
        target='bayesnetcpd')

    bld(features='c cshlib cxx pyext',
        source=['samcnet/probability.pyx'],
        #libpath=['.','./build'],
        includes=['include'],
        cxxflags=CXXFLAGS,
        ldflags=LDFLAGS,
        target='probability')

    CFLAGS.remove('-Wall')
    libs = ['MATH', 'DAI', 'GMP', 'GMPXX']
    
    bld(features='c cshlib cxx pyext',
        source=['samcnet/pydai.pyx'],
        libpath=['lib'],
        includes=['include'],
        use=libs,
        cxxflags=CXXFLAGS,
        ldflags=LDFLAGS,
        target='pydai')

    #bld.env['PREFIX'] = '.'
    #for x in 'dai.so probability.so bayesnetcpd.so bayesnet.so samc.so cost.so'.split():
        #bld.symlink_as('${PREFIX}/lib/%s' % x, 'build/%s' % x)

def dist(ctx):
        ctx.excl      = '**/*.zip **/*.bz2 **/.waf-1* **/*~ **/*.pyc **/*.swp **/.lock-w*' 
        #ctx.files     = ctx.path.ant_glob('**/wscript **/*.h **/*.cpp waf') 


## SAMCNet

This package started as a toolkit and demonstration of Bayesian model averaging 
applied to a class of graphical models known as Bayesian networks. I then added 
functionality to perform optimal Bayesian Classification for a publication 
[[Knight, Ivanov, Dougherty 
2014]](http://www.biomedcentral.com/bmcbioinformatics/mostviewed/30days).
In other words, it can handle classification of RNA-Seq data using a the 
published statistical model that shows superior performance when compared to 
nonlinear SVM, LDA, and others. 

Both of these functionalities still work, although for cutting edge 
development, effort has moved over to the Julia ports for classification 
[(OBC.jl)](https://github.com/binarybana/OBC.jl), network inference
[(MCBN.jl)](https://github.com/binarybana/MCBN.jl), and a package split off to 
contain the MCMC methods at the API resolution I needed 
[(SAMC.jl)](https://github.com/binarybana/SAMC.jl).

## Installing
In a recent version of Ubuntu you'll need the following:
```
sudo apt-get install cython python-pandas python-numpy python-scipy 
python-networkx libboost-dev libboost-program-options-dev libboost-test-dev 
libjudy-dev libgmp-dev
git clone git@github.com:binarybana/samcnet.git
git submodule update --init
cd deps/libdai
make -j
cd ../..
ln -s ../deps/libdai/lib/libdai.so lib/
for f in build/*.so; ln -s ../$f samcnet/; done
./waf configure
./waf
export LD_LIBRARY_PATH=lib:build
```

Then test with
```
python -m tests.test_net
```

### Usage

A video tutorial explaining how to operate the classifier on your RNA-Seq 
dataset has been posted at: http://www.youtube.com/watch?v=fPa5qy1tdhY

## Building Blocks

This software would not be possible without the following components:
- Python for the main driving and glue code
- Cython for C and C++ integration and speed
- [libdai](http://cs.ru.nl/~jorism/libDAI/) for Bayesian network inference.
- [Redis](http://redis.io) for the (optional) distributed job management
- [waf](http://code.google.com/p/waf/) for the build system
- rsyslog for remote logging


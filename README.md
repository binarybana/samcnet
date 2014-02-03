## SAMCNet

SAMCNet is a toolkit and demonstration for Bayesian model averaging over 
objective functions defined over model classes of interest.

Particularly, it currently handles classification of RNA-Seq data using a 
(submitted) statistical model that shows good performance in comparison to 
SVM, LDA, and others. Video tutorials explaining how to operate the 
classifier on your RNA-Seq dataset will be posted here in short order.

Also, we can use a model class of discrete, static Bayesian 
networks defined over the variables of interest. Then we can use objective functions 
to simplify the complex posterior over this large model class.

I'll go more into detail regarding the theory, code, and applications once the paper
is published.

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

## Building Blocks

This software would not be possible without the following components:
- Python for the main driving and glue code
- Cython for C and C++ integration and speed
- [libdai](http://cs.ru.nl/~jorism/libDAI/) for Bayesian network inference.
- [Redis](http://redis.io) for the (optional) distributed job management
- [waf](http://code.google.com/p/waf/) for the build system
- rsyslog for remote logging


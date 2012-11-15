## SAMCNet

SAMCNet is a toolkit and demonstration for Bayesian model averaging over 
objective functions defined over model classes of interest.

Specifically, we are here using the model class of discrete, static Bayesian 
networks defined over the variables of interest, and using objective functions 
to simplify the complex posterior over this large model class.

I'll go more into detail regarding the theory, code, and applications once I 
have some papers published.

## Building Blocks

This software would not be possible without the following components:
- Python for the main driving and glue code
- Cython for C and C++ integration and speed
- [libdai](http://cs.ru.nl/~jorism/libDAI/) for Bayesian network inference.
- [Redis](http://redis.io) for the distributed job management
- [waf](http://code.google.com/p/waf/) for the build system
- rsyslog for remote logging
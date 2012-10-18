#include <string>
#include <dai/factorgraph.h>
#include <dai/factor.h>
#include <dai/varset.h>

#ifndef __utils__h
#define __utils__h

std::string crepr(const dai::FactorGraph &x);
std::string crepr(const dai::Factor &x);
std::string crepr(const dai::VarSet &x);

#endif

#include <iostream>
#include <string>
#include <sstream>
#include <dai/factorgraph.h>
#include <dai/factor.h>
#include <dai/varset.h>

using namespace std;

string crepr(const dai::FactorGraph &x) {
    ostringstream s;
    s << x;
    return s.str();
}

string crepr(const dai::Factor &x) {
    ostringstream s;
    s << x;
    return s.str();
}

string crepr(const dai::VarSet &x) {
    ostringstream s;
    s << x;
    return s.str();
}

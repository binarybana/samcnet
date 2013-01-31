import numpy as np
import scipy.stats as st

class Simple():
    def __init__(self, truemu=2.0, sigma=1.0, N=20, mu0=0.0, tau=5.0):
        self.x = 0.0
        self.old_x = 0.0

        self.truemu = truemu
        self.sigma = sigma
        self.N = N

        self.mu0 = mu0
        self.tau = tau

        self.data = st.norm.rvs(loc=self.truemu, scale=self.sigma, size=self.N)

        self.postmode = (self.mu0/self.tau**2 + self.data.mean()/self.sigma**2) \
                / (1/self.tau**2 + 1/self.sigma**2)
            
        print "Postmode: %f" % self.postmode

    def propose(self):
        self.old_x = self.x
        self.x += np.random.normal(scale=3.0)

    def copy(self):
        return self.x

    def save_to_db(self):
        return self.x

    def reject(self):
        self.x = self.old_x

    def energy(self):
        sum = 0.0
        # prior
        sum -= st.norm.logpdf(self.x, loc=self.mu0, scale=self.tau)

        # likelihood
        sum -= st.norm.logpdf(self.data, loc=self.x, scale=self.sigma).sum()
        return sum

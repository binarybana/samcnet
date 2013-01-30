
class MHRun():
    def __init__(self, obj, burn, thin=1):
        self.obj = obj
        self.burn = burn
        self.db = None

        self.mapvalue = None
        self.mapenergy = None

        self.thin = thin

        self.iteration = 0

        self.propose = 0
        self.accept = 0

    def sample(self, num):
        num = int(num)
        self.db = self.obj.init_db(self.db, (self.iteration + num - self.burn)//self.thin, 'mh')
        minenergy = np.infty

        oldenergy = self.obj.energy()
        for i in range(int(num)):
            self.iteration += 1

            self.obj.propose()
            self.propose+=1

            newenergy = self.obj.energy()

            r = oldenergy - newenergy # ignoring non-symmetric proposals for now
            if r > 0.0 or np.random.rand() < exp(r):
                # Accept
                oldenergy = newenergy
                self.accept += 1
            else:
                self.obj.reject()

            if self.iteration>self.burn and i%self.thin == 0:
                self.obj.save_to_db(self.db, 0, oldenergy, (self.iteration-self.burn-1)//self.thin)

            if oldenergy < minenergy:
                minenergy = oldenergy
                self.mapvalue = self.obj.copy()
                self.mapenergy = oldenergy

            if self.iteration%1e3 == 0:
                print "Iteration: %9d, best energy: %7f, current energy: %7f" \
                        % (self.iteration, minenergy, oldenergy)

        print "Sampling done, acceptance: %d/%d = %f" \
                % (self.accept, self.propose, float(self.accept)/float(self.propose))
            

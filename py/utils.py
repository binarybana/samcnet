import pylab as p
import os
import networkx as nx
import pandas as pa
import pebl as pb
import StringIO as si

def plotHist(s):
    p.subplot(211)
    p.plot(s.hist[0], s.hist[1], 'go')
    p.title('Energy - theta')
    p.subplot(212)
    p.plot(s.hist[0], s.hist[2], 'go')
    p.title('Sample Counts')
    
def drawGraph(graph):
    fname = os.tempnam()
    nx.write_dot(graph, fname+'.dot')
    os.popen('dot -Tsvg -o %s.svg %s.dot' % (fname,fname))
    os.popen('xdg-open %s.svg > /dev/null' % fname)

def before_and_after(g1, g2):
    ag1, ag2 = nx.to_agraph(g1), nx.to_agraph(g2)

    ag1.layout(prog='dot')
    ag1.draw('/tmp/f1.svg')
    ag1.delete_edges_from(ag1.edges())
    ag1.add_edges_from(ag2.edges())
    ag1.draw('/tmp/f2.svg')
    os.popen('xdg-open /tmp/f1.svg > /dev/null')
    os.popen('xdg-open /tmp/f2.svg > /dev/null')

def to_pebl(states, data):
    header = ['%d,discrete(%d)' %(i,a) for i,a in enumerate(states)]
    df = pa.DataFrame(data, columns=header)
    x = si.StringIO()
    x.write('\t'.join(header) + '\n')
    df.to_csv(x, header=False, index=True, sep='\t')
    x.seek(0)
    return pb.data.fromstring(x.read())

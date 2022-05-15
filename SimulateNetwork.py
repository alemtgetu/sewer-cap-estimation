
import networkx as nx
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
# flow dynamics on the tree


class FlowNetwork:

    def __init__(self, design_list, r_crit):
        self.design_list = design_list
        self.ntiers = len(design_list)
        self.r_crit = r_crit
        self.L = len(design_list)
        G, edges, nodes, n2e = self.make_tree(design_list)
        self.G = G.copy()
        self.edges = edges
        self.nodes = nodes
        self.n2e = n2e
        self.N = len(nodes)
        numberOfInnerNodes = self.sum_nodes_by_layer(design_list[:self.L-1])
        self.topNodes = [i for i in range(self.N) if i >= numberOfInnerNodes]
        self.NT = len(self.topNodes)
        self.NI = self.N-self.NT
        self.A = nx.adjacency_matrix(G)
        self.mincap = self.makeMinCap()
        mult = np.random.uniform(1.1, 1.5, self.N-1)
        mult = np.insert(mult, 8, 1)
        self.cap = mult*self.mincap.flatten()

    def getMinCap(self): return self.mincap.flatten()
    def getCap(self): return self.cap.flatten()

    def sum_nodes_by_layer(self, ns):
        s = 0
        for i in range(len(ns)+1):
            s += np.prod(ns[0:i])
        return s

    def custom_tree_edges(self, tiers):
        n, m, M, edges = 0, 1, 1, [(0, 0)]
        for t in tiers:
            for i in range(n, M):
                n += 1
                for j in range(t):
                    m += 1
                    e = (m-1, n-1)
                    edges.append(e)
            M = m
        return edges

    def printv(self, msg, x, verbose=False):
        if verbose:
            print(msg, '_'.join([str(int(i)) for i in x]))

    def makeMinCap(self):
        v = False
        x0 = np.append(np.zeros(self.NI), self.r_crit +
                       np.ones(self.NT)).reshape((self.N, 1))
        self.printv('', x0.flatten(), v)
        x = x0.copy()
        xi = x0.copy()
        for i in range(self.ntiers):
            xi = self.A.T.dot(xi)
            self.printv('', xi.flatten(), v)
            x += xi
        return (x.T)

    def make_tree(self, design_list):
        edges = self.custom_tree_edges(design_list)
        G = nx.DiGraph()
        for e in edges:
            G.add_edge(e[0], e[1])
        G.remove_edge(0, 0)
        edges = G.edges()
        nodes = G.nodes()
        n2e = {}
        for n in nodes:
            for e in edges:
                if e[0] == n:
                    n2e[n] = e
        return([G, edges, nodes, n2e])

    def regulateOverflowAndLoad(self, flow_demand, retained_load_demand, givenCaps):
        if givenCaps is not None:
            pass
        else:
            givenCaps = self.cap
        flow_demand = flow_demand.flatten()
        retained_load_demand = retained_load_demand.flatten()
        regulated_flow = np.array([
            flow_demand[i]
            if flow_demand[i] <= givenCaps[i] else givenCaps[i]
            for i in range(len(flow_demand))
        ])
        overflow = np.array([
            flow_demand[i]-givenCaps[i]
            if flow_demand[i] >= givenCaps[i] else 0
            for i in range(len(flow_demand))
        ])
        delta = np.array([
            regulated_flow[i]/flow_demand[i]
            if flow_demand[i] > 0 else 1
            for i in range(len(flow_demand))
        ])
        retained_load = np.array([
            retained_load_demand[i]*delta[i] for i in range(len(flow_demand))
        ])

        return regulated_flow, overflow, retained_load

    def makeFlow(self, rain, givenCaps):
        input_flow_demand = np.append(np.zeros(self.NI), np.ones(self.NT) + rain
                                      ).reshape((self.N, 1))  # ;self.printv('',input_flow_demand.flatten(),'False')
        accumulated_overflow = np.zeros(self.N)
        accumulated_regulated_flow = np.zeros(self.N)
        accumulated_retained_load = np.zeros(self.N)
        retained_load_demand = np.append(np.zeros(self.NI), np.ones(self.NT))
        for tier in range(self.ntiers+1):
            regulated_flow_i, overflow_i, retained_load_i = self.regulateOverflowAndLoad(
                input_flow_demand, retained_load_demand, givenCaps
            )
            accumulated_regulated_flow += regulated_flow_i
            accumulated_overflow += overflow_i
            accumulated_retained_load += retained_load_i
            # the adjacency matrix pushes the flow & reteained load forward to next set of nodes
            input_flow_demand = self.A.T.dot(regulated_flow_i)
            retained_load_demand = self.A.T.dot(retained_load_i)

        return (accumulated_regulated_flow, accumulated_overflow, accumulated_retained_load)

    def reportFlowOverflowAndLoad(self, r, givenCaps=None):
        accumulated_regulated_flow, accumulated_overflow, accumulated_retained_load = self.makeFlow(
            r, givenCaps)
        ii = np.where(accumulated_overflow > 0)[0]
        ofl = [g for g in accumulated_overflow if g > 0]
        return (ii, ofl, accumulated_regulated_flow, accumulated_retained_load)

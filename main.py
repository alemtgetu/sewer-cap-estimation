
from networkx.drawing.nx_pydot import graphviz_layout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from SimulateNetwork import FlowNetwork
import networkx as nx
from typing import Tuple
# import pydot
# import sys

do_test = False
if do_test:
    fn = FlowNetwork([3, 4, 3, 2], 2.2)
    # print(fn.A)
    # print(fn.A.toarray())
    ofr = fn.reportFlowOverflowAndLoad(3.7)
    print(ofr)
    exit(0)
#print("overflow nodes with rain=",r1,":",ofl_nodes2)
#print("overflow amounts with rain=",r1,"ofl=",ofl_amts2)


def showTree(fn, overflow_report, colors):
    pos = graphviz_layout(fn.G, prog="twopi")
    plt.figure(figsize=(10, 10))
    nx.draw(fn.G, pos, with_labels=True, font_color="black",
            font_size=12, label="no overflow")
    for r in sorted(overflow_report.keys(), reverse=True):
        nx.draw_networkx_nodes(
            fn.G, pos, overflow_report[r]['ofl_nodes'], node_color=colors[r], label="overflow at rain="+str(r))
    #nx.draw_networkx_nodes(G, pos, ofl_nodes3, node_color="gray",label="overflow at rain="+str(r3))
    #nx.draw_networkx_nodes(G, pos, ofl_nodes2, node_color="orange",label="overflow at rain="+str(r2))
    #nx.draw_networkx_nodes(G, pos, ofl_nodes1, node_color="red",label="overflow at rain="+str(r1))
    plt.title("Network structure: "+str(fn.design_list) +
              " critical rain level ="+str(fn.r_crit)+" and d = 1")
    plt.legend()
    plt.show()


def doAnalysis(protocols, show_tree) -> Tuple[dict, FlowNetwork]:
    """creates flow network and gets the overflow reports

    Parameters
    ----------
    protocols: dictionary
    The flow network protocl as dictionary
        keys:
        design_list: list  
            first element - number of feeding nodes to the central node
            second element - number of teirs the flow graph will have
            third element - the number of feeding nodes for each nodes in the first teir
            fourth element - (if applicable) the number of nodes at the edge (last teir)
        colors: dictionary
            overflow color indicators per rain level 
        r_critical: int
            the rain critical amount
    show_tree: bool
        If True the simulated flow newtork graph tree will be displayed
    Returns
    -------
    overflow_report: dict
        for each rain level as key
        ofl_nodes, ofl_amounts, arf (accumlated regulated flow), load (bacterial load amount at each node)]
    fn: FlowNetwork
    """

    overflow_reports = []
    # for pr in protocols.keys():
    fn = FlowNetwork(protocols['design_list'], protocols['r_crit'])
    # fns.append(fn)
    overflow_report = {}
    # colors = protocols[1]['colors']
    for r in protocols['colors'].keys():
        ofl_nodes, ofl_amts, arf, load = fn.reportFlowOverflowAndLoad(r)
        overflow_report[r] = {}
        overflow_report[r]['ofl_nodes'] = ofl_nodes
        overflow_report[r]['ofl_amts'] = ofl_amts
        overflow_report[r]['arf'] = arf
        overflow_report[r]['load'] = load
    overflow_reports.append(overflow_report)
    if show_tree:
        showTree(fn, overflow_report, protocols['colors'])
    return overflow_reports, fn


def max_entropy(fG, totCap, fn, overflow_report, testMe=False):
    verbose = testMe
    estcap = cvx.Variable(fn.N)  # from 1..N ignore 0
    # sum over i of x[i] log [ x[i] ]
    objective = cvx.Maximize(cvx.sum(cvx.entr(estcap)))

    constraints = [0 <= estcap, estcap <= totCap, sum(estcap) == totCap
                   ]
    # at this point the solution is estcap[i] = totalCap/N for all i

    # at critical rain:
    constraints.append(estcap >= fn.getMinCap())

    for r in overflow_report.keys():
        ofl_amts = overflow_report[r]['ofl_amts']
        ofl_nodes = overflow_report[r]['ofl_nodes']
        num_ofl_nodes = len(ofl_nodes)
        for i in range(num_ofl_nodes):  # overflow constraints
            x = fG.neighbors(ofl_nodes[i])
            parents = [k for k in x if k > ofl_nodes[i]]
            if len(parents) > 0:  # if internal node
                # flow_demand - flow = overflow
                # flow[a] + flow[b] - flow[i] = overflow
                # [ 1, -1, -1, -1] * [ k_i, k_a, k_b, k_c] <= -w_i
                y = [1]
                y.extend([-1]*len(parents))
                y = np.array(y)
                yy = [ofl_nodes[i]]
                yy.extend(parents)
                yy = np.array(yy)
                if verbose:
                    print("OF: ", y, ".", yy, "<=", -ofl_amts[i])
#                constraints.append(y @ estcap[yy-1] <= -ofl_amts[i]) #possibly right
                # probably right
                constraints.append(y @ estcap[yy] <= -ofl_amts[i])
            else:  # if toplevel node
                yy = [ofl_nodes[i]]
                if verbose:
                    print("OF: ", yy, "<=", r+1-ofl_amts[i])
                constraints.append(estcap[ofl_nodes[i]] <= r+1-ofl_amts[i])
        for i in range(fn.N-1):  # non overflow constraints
            if i not in ofl_nodes:
                x = fG.neighbors(i)
                parents = [k for k in x if k > i]
                if len(parents) > 0:  # no nonoverflow constraints for internal nodes
                    # no generalization possible here; not true that est_capacity[i]> sum est_capacity of parents
                    pass
                else:  # toplevel node
                    yy = [i]
                    if verbose:
                        print("normal [1].", yy, ">=", r+1)
                    constraints.append(estcap[i] >= r+1)

    constraints.append(estcap[0] >= 300)  # 270)
    # at r2
    prob = cvx.Problem(objective, constraints)
    result = prob.solve()
    if testMe:
        print(result)
    print("EstCap", estcap)
    return(estcap.value)


def main() -> None:

    protocols = {}
    colors = {}
    colors[2.2] = "gray"
    colors[2.21] = "red"
    colors[2.4] = "orange"
    colors[2.6] = "yellow"
    colors[2.8] = "green"
    colors[3.0] = "cyan"
    colors[3.5] = "purple"
    protocols["colors"] = colors
    # protocols["design_list"] = [2, 2, 2]
    protocols["design_list"] = [3, 4, 3, 2]
    protocols["r_crit"] = 2.2

    np.random.seed(23)
    # print(protocols)
    overflow_reports, fn = doAnalysis(protocols, False)
    # print("overlow report", overflow_reports, sep="\n")

    # prnum = 0
    totMinCap = np.sum(fn.mincap[0])
    totCap = 1.13*totMinCap
    fG = nx.Graph(fn.G)
    cap_maxent = max_entropy(
        fG, totCap, fn, overflow_reports[0], False)
    # print(cap_maxent)
    # return 0
    print("The maxent estimate of flow capacities is . . . \n",
          np.around(cap_maxent, 2))
    # cap_maxent-fns[prnum].cap


main()

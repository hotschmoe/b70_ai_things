import os, sys
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); sys.path.insert(0, D + "/../shared")
from check import check, finish
import cbsgen
L = "/home/pat/logs/"
def ans():
    lines = open(L + "answer.txt").read().strip("\n").split("\n")
    return dict(l.split("=", 1) for l in lines), lines
check("four_lines_exact_keys", lambda: [l.split("=")[0] for l in ans()[1]] == list(cbsgen.ANSWER))
for k, v in cbsgen.ANSWER.items():
    check("answer_" + k, lambda k=k, v=v: ans()[0][k] == v)
check("safe_log_untouched", lambda: open(L + "CBS.log").read() == cbsgen.build())
finish()

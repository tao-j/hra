import numpy as np
import glob
import io
from contextlib import redirect_stdout
'''
\\begin{table}[t]
    \\centering
    \\caption{Number of samples requested by respective algorithms when the sorting is ended.
    }
    \\begin{small}
    \\begin{tabular}{cccccc}
      \\toprule
      \\multirow{2}{*}{\\shortstack[t]{}}&\\multirow{2}{*}{$\\gamma_B$}&\\multirow{2}{*}{Methods}&\\multicolumn{3}{c}{$\\gamma_A$}\\\\
      \\cmidrule(lr){4-6}
      &&& 2.5 & 5 & 10  \\\\
      \\midrule

\\multirow{ 6 }{*}{}
&\\multirow{2}{*}{0.25} &Baseline& \\small{31734}\\small{$\\pm$}\\small{3365}&\\small{27669}\\small{$\\pm$}\\small{1851}&\\small{19254}\\small{$\\pm$}\\small{785} \\\\ &&Elimination& \\small{26629}\\small{$\\pm$}\\small{3039}&\\small{20032}\\small{$\\pm$}\\small{1092}&\\small{11156}\\small{$\\pm$}\\small{960} \\\\ \\cmidrule(lr){2-6}
&\\multirow{2}{*}{1.0} &Baseline& \\small{28258}\\small{$\\pm$}\\small{1109}&\\small{21706}\\small{$\\pm$}\\small{632}&\\small{16390}\\small{$\\pm$}\\small{1120} \\\\ &&Elimination& \\small{27569}\\small{$\\pm$}\\small{998}&\\small{18074}\\small{$\\pm$}\\small{1588}&\\small{11356}\\small{$\\pm$}\\small{706} \\\\ \\cmidrule(lr){2-6}
&\\multirow{2}{*}{2.5} &Baseline& \\small{19272}\\small{$\\pm$}\\small{1260}&\\small{16702}\\small{$\\pm$}\\small{779}&\\small{9664}\\small{$\\pm$}\\small{1083} \\\\ &&Elimination& \\small{18950}\\small{$\\pm$}\\small{1133}&\\small{16797}\\small{$\\pm$}\\small{663}&\\small{9046}\\small{$\\pm$}\\small{1204} \\\\ \\bottomrule

\\end{tabular}\\end{small}\\end{table}
'''

avgs = []
stds = []

lines = open("exp.txt").readlines()
gb = ["0.25", "1.0", "2.5"]
for i in range(3):
    print("&\\multirow{{2}}{{*}}{{ {} }}".format(gb[i]), "&Baseline", lines[i].strip('\n') + " \\\\ &&Elimination" + lines[i + 4].strip('\n'), "\\\\ \\cmidrule(lr){2-6}")

import numpy as np
import glob
import io
from contextlib import redirect_stdout


def txt2tex(all, dist):
    all_lines = all.splitlines()
    assert (len(all_lines) == 36)

    tex_str = ""
    with io.StringIO() as buf, redirect_stdout(buf):
        for ll in [3, 2, 1, 0]:
            st = ll * 9
            print("\\multirow{{9}}{{*}}{{$\\{}$}}".format(all_lines[st].strip()))
            lines = list(map(str.strip, all_lines[st + 1:st + 8]))
            pp = []
            pn = []
            for l in lines[:3]:
                pp.append(list(map(lambda x: "\\small{" + "{:0.3f}".format(float(x)) + "}", l.split())))
                pn.append(list(map(lambda x: float("{:0.3f}".format(float(x))), l.split())))
            for l in lines[4:7]:
                pp.append(list(map(lambda x: "\\small{" + "{:0.3f}".format(float(x)) + "}", l.split())))

            pn = np.array(pn[:3])
            max_idx = np.argmax(pn[:3], axis=0)
            for i in range(9):
                sel = pn[:, i] == pn[max_idx[i]][i]
                if np.sum(sel) > 1:
                    for sel_i, sel_flag in enumerate(sel):
                        if sel_flag:
                            pp[sel_i][i] = "\\underline{" + pp[sel_i][i] + "}"
                else:
                    for sel_i, sel_flag in enumerate(sel):
                        if sel_flag:
                            pp[sel_i][i] = "\\textbf{" + pp[sel_i][i] + "}"

            # sort_idx = np.argsort(-pn[:3], axis=0)
            # for i in range(9):
            #     n1, n2 = sort_idx[0:2, i]
            #     if (pn[n1][i] - pn[n2][i]) > 0.004:
            #         pp[n1][i] = "\\textbf{" + pp[n1][i] + "}"

            line_p = []
            for i in range(3):
                f_pm = lambda x: "\\small{$\\pm$}".join(x)
                line_tokens = list(map(f_pm, zip(pp[i], pp[i + 3])))
                line_p.append(line_tokens)

            gamma_bs = [0.25, 1.0, 2.5]
            rules = ["\\cmidrule(lr){2-6}", "\\cmidrule(lr){2-6}", "\midrule"]

            if dist == 'g':
                a, b, c = ["BTL", "CrowdBT", "HBTL"]
            if dist == 'n':
                a, b, c = ["TCV", "CrowdTCV", "HTCV"]
            fmt = f"&{a}& {{}} \\\\ &&{b}& {{}} \\\\ &&{c}& {{}} \\\\"
            for i in range(3):
                this_line = fmt.format(
                    *["&".join(line_p[k][i::3]) for k in range(3)])
                this_line += rules[i]
                gamma_b = gamma_bs[i]
                print(f"&\multirow{{3}}{{*}}{{{gamma_b}}} " + this_line)

            print("")

        tex_str = buf.getvalue()

    return tex_str


if __name__ == "__main__" or True:
    files = glob.glob("dump*.txt")
    for in_f in files:
        dist = in_f.split('_')[1][0]
        assert(dist in ['g', 'n'])

        with open(in_f) as in_ff:
            tex_str = txt2tex(in_ff.read(), dist)
            out_f = open(in_f.replace("txt", "tex"), 'w')
            out_f.write(tex_str)
            out_f.close()


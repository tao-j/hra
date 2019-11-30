import numpy as np

for nn in ['20', '25', '50', '80', '99']:
    print('-------------------------------', nn)
    lines = open('cp16/{}.txt'.format(nn)).readlines()
    stats_acc = {}
    stats_ken = {}
    for line in lines:
        name, acc, ken = line.split(',')
        if name not in stats_acc:
            stats_acc[name] = []
        stats_acc[name].append([float(acc), float(ken)])

    for k, v in stats_acc.items():
        print(k, np.average(np.array(v), axis=0), np.std(np.array(v), axis=0))
        
import pandas as pd

#%%
df = pd.read_csv('data/galleryimage/data_anonymised_images.csv')

lines = []
for i in range(1, 10):
    for j in range(i+1, 11):
        col = df['pairwise_{}_{}'.format(i, j)]
        for idx, res in col.items():
            # 1 means win
            if res == 1:
                line = '{} {} {}'.format(idx+1, i, j)
            elif res == 2:
                line = '{} {} {}'.format(idx+1, j, i)
            else:
                raise NotImplementedError
            lines.append(line)

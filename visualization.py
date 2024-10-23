import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.style.use('seaborn-v0_8-darkgrid')

def txt_to_png(path : str):
    return path.replace('heat', 'grid').replace('.txt', '.png')



block_size = [2, 4, 8, 16, 32]
times = [15.74, 5.66, 1.79, 1.5, 1.7]
sns.scatterplot(x = block_size, y = times)
sns.lineplot(x = block_size, y = times)
plt.xlabel('Block size')
plt.ylabel('Time (s)')
plt.title('Execution time vs Block_Size')
plt.savefig('Visualization/ExecTimes_vs_BlockSize.png')
plt.clf()



grids_folder = 'Grids_text/'

for txt_file in os.listdir(grids_folder):
    grid = np.loadtxt(grids_folder + txt_file, usecols = range(100))
    plt.figure(figsize = (10, 8))
    sns.heatmap(grid)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'Visualization/{txt_to_png(txt_file)}')
    plt.clf()


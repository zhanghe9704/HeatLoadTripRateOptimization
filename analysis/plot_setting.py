import matplotlib.pyplot as plt
# set global settings for plotting
def init_plotting():
    plt.rcParams['figure.figsize'] = (8, 8)  # 1/4 of line width
    # plt.rcParams['figure.figsize'] = (5.6, 2.6)  # 1/4 of line width
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'CM Sans Serif'
    #    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['text.usetex'] = True
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    #    plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.loc'] = 'center left'
    plt.rcParams['legend.numpoints'] = 1
    plt.rcParams['axes.linewidth'] = 1
    #    plt.gca().spines['right'].set_color('none')
    #    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
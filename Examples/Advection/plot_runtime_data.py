import matplotlib as mpl
# Use the pgf backend (must be set before pyplot imported)
mpl.use('pgf')

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Use the seborn style
plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False     # don't setup fonts from rc parameters
    })

def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    #golden_ratio = (5**.5 - 1) / 2
    height_ratio = 0.75

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * height_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


mb_DG = np.loadtxt("8c_DG.txt")
mb_FV = np.loadtxt("8c_FV.txt")
ws_DG = np.loadtxt("18c_DG.txt")
ws_FV = np.loadtxt("18c_FV.txt")
sr_DG = np.loadtxt("128c_DG.txt")
sr_FV = np.loadtxt("128c_FV.txt")

data_DG = [
    mb_DG,
    ws_DG,
    sr_DG,
]

data_FV = [
    mb_FV,
    ws_FV,
    sr_FV,
]

significant_combinations = []
# The indices of the two populations that should be compared
# We take FV and DG from each machine
combinations = [(0,0),(1,1),(2,2)]

for combination in combinations:
    data1 = data_DG[combination[0]]
    data2 = data_FV[combination[1]]
    # Significance
    U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    if p < 0.05:
        significant_combinations.append([combination, p])

ticks = ['8', '18', '128']

# function for setting the colors of the box plots pairs
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

fig, ax = plt.subplots(1,1,figsize=set_size(345,fraction=0.7))
bp1 = ax.boxplot(data_DG, positions=np.array(range(len(data_DG)))*2.0-0.4,
                  showfliers=False, widths=0.6)
set_box_color(bp1,'#D7191C')
bp2 = ax.boxplot(data_FV, positions=np.array(range(len(data_FV)))*2.0+0.4, 
                 showfliers=False, widths=0.6)
set_box_color(bp2,'#2C7BB6')


ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['DGM', 'FVM'],
           loc='upper left')
ax.set_xlabel("Number of Processors")
ax.set_ylabel("Wall time per time step [s]")
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
ax.vlines([1.,3.],ymin=0,ymax=0.5,colors='grey')

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-1, (len(ticks)*2)-1)
plt.ylim(0.01, 0.045)
plt.tight_layout()
#plt.show()

# Get the y-axis limits
bottom, top = ax.get_ylim()
y_range = top - bottom

# Significance bars
for i, significant_combination in enumerate(significant_combinations):
    # Columns corresponding to the datasets of interest
    x1 = np.max(bp1["whiskers"][2*i].get_xdata())
    x2 = np.max(bp2["whiskers"][2*i].get_xdata())
    # Plot the bar
    mean_DG = np.mean(data_DG[i])
    mean_FV = np.mean(data_FV[i])
    whisker_height_l = np.max(bp1["caps"][2*i+1].get_ydata())
    whisker_height_r = np.max(bp2["caps"][2*i+1].get_ydata())
    bar_height = np.maximum(mean_DG,mean_FV) + 0.005
    bar_tip_l = whisker_height_l + 0.001
    bar_tip_r = whisker_height_r + 0.001
    bar_tips = bar_height - (y_range * 0.04)
    plt.plot(
        [x1, x1, x2, x2],
        [bar_tip_l, bar_height, bar_height, bar_tip_r], lw=1, c='k'
    )
    # Significance level
    p = significant_combination[1]
    if p < 0.0001:
        sig_symbol = '****'
    elif p < 0.001:
        sig_symbol = '***'
    elif p < 0.01:
        sig_symbol = '**'
    elif p < 0.05:
        sig_symbol = '*'
    text_height = bar_height + (y_range * 0.01)
    plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')

plt.savefig('figure.pgf', format='pgf')

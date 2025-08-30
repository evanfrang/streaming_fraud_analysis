import matplotlib as mpl # type: ignore
import matplotlib.pyplot as plt # type: ignore

plt.style.use('bmh')

CUSTOM_RCPARAMS = {
    "text.usetex": False,
    "axes.linewidth": 2,
    "font.family": "serif",
    "font.weight": "bold",
    #"text.latex.preamble": r"\usepackage{sfmath} \boldmath",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 20,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 14,
}

# Function to apply settings
def apply_rcparams():
    mpl.rcParams.update(CUSTOM_RCPARAMS)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use('seaborn-darkgrid')

# Use TkAgg backend for matplotlib
plt.switch_backend('TkAgg')

interval = 500

def animate(i):
    data = pd.read_csv('loss.csv', header=None, names=['Training Loss', 'Validation Loss'])
    data = data.apply(pd.to_numeric, errors='coerce').fillna(method='ffill')

    plt.clf()  

    if len(data) > interval:
        data_to_plot = data[-interval:]
    else:
        data_to_plot = data
    plt.plot(data_to_plot['Training Loss'], label='Training Loss', color='blue')
    plt.plot(data_to_plot['Validation Loss'], label='Validation Loss', color='red')
    plt.legend(loc='upper right')
    plt.tight_layout()

    ax_inset = inset_axes(plt.gca(), width="30%", height="30%", loc='upper right')
    ax_inset.plot(data['Training Loss'], color='blue', alpha=0.5)
    ax_inset.plot(data['Validation Loss'], color='red', alpha=0.5)
    # Remove axis numbers
    ax_inset.set_xticklabels([])
    ax_inset.set_yticklabels([])

    plt.autoscale(axis='y')
    ax_inset.autoscale(axis='y')

ani = FuncAnimation(plt.gcf(), animate, interval=interval)

plt.tight_layout()

# Retrieve the Tkinter canvas window and set it to stay on top
plt.gcf().canvas.manager.window.attributes('-topmost', 1)

plt.show()
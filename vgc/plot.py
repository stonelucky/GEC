import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
# %matplotlib inline

flags = tf.app.flags
FLAGS = flags.FLAGS
def plot():
    evals_name = 'ACC=, f1_macro=, precision_macro=, recall_macro=, f1_micro=, precision_micro=, recall_micro=, NMI=, ADJ_RAND_SCORE='.strip('=').split('=,')

    df = pd.read_csv('recoder.txt', sep=',', header=None, names=evals_name)
    for name in evals_name:
        df[name] = [float(item.replace(name+'=', '')) for item in df[name]]

    df.plot.line()
    plt.legend(loc=3,ncol=3, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('figure_cvae_cora.pdf', )           
    plt.show()

plot()
# fig = df[0].get_figure()


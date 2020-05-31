import settings
import tensorflow as tf
import numpy as np

from clustering import Clustering_Runner
from link_prediction import Link_pred_Runner
from visualization import Visualization_Runner
import time

start = time.clock()

dataname = 'citeseer'       # {'cora', 'citeseer', 'pubmed', 'blog', 'flickr', 'cornell', 'texas', 'washington', 'wisconsin'}
model = 'gae'          # 'gae' or 'vgae' or 'arga' or 'arvga' or 'vgc' or 'vgcg'
task = 'visualization'         # 'clustering' or 'link_prediction' or 'visualization'

settings_ = settings.get_settings(dataname, model, task)

if __name__ == '__main__':
    if task == 'clustering':
        runner = Clustering_Runner(settings_)
    elif task == 'visualization':
        runner = Visualization_Runner(settings_)
    else:
        runner = Link_pred_Runner(settings_)


    times = 1

    for i in range(times):
        tf.reset_default_graph()
        runner.erun()

end = time.clock()

print('Epochs: {}, Running time: {} Seconds, while using {} on {}'.format(settings_['iterations'], end-start, model, dataname))

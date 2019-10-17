from utils.Dataset import Dataset
from model import model
from utils.print_result import print_result
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
each net has its own learning_rate(lr_xx), activation_function(act_xx), nodes_of_layers(dims_xx)
ae net need pretraining before the whole optimization
'''
if __name__ == '__main__':
    data = Dataset('handwritten_2views')
    x1, x2, gt = data.load_data()
    x1 = data.normalize(x1, 0)
    x2 = data.normalize(x2, 0)
    n_clusters = len(set(gt))

    act_ae1, act_ae2, act_dg1, act_dg2 = 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'
    dims_ae1 = [240, 200]
    dims_ae2 = [216, 200]
    dims_dg1 = [64, 200]
    dims_dg2 = [64, 200]

    para_lambda = 1
    batch_size = 100
    lr_pre = 1.0e-3
    lr_ae = 1.0e-3
    lr_dg = 1.0e-3
    lr_h = 1.0e-1
    epochs_pre = 10
    epochs_total = 20
    act = [act_ae1, act_ae2, act_dg1, act_dg2]
    dims = [dims_ae1, dims_ae2, dims_dg1, dims_dg2]
    lr = [lr_pre, lr_ae, lr_dg, lr_h]
    epochs_h = 50
    epochs = [epochs_pre, epochs_total, epochs_h]

    H, gt = model(x1, x2, gt, para_lambda, dims, act, lr, epochs, batch_size)
    print_result(n_clusters, H, gt)

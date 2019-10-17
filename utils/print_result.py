from utils.cluster import cluster
import warnings

warnings.filterwarnings('ignore')


def print_result(n_clusters, H, gt, count=10):
    acc_avg, acc_std, nmi_avg, nmi_std, ri_avg, ri_std, f1_avg, f1_std = cluster(n_clusters, H, gt, count=count)
    print('clustering h      : acc = {:.4f}, nmi = {:.4f}'.format(acc_avg, nmi_avg))

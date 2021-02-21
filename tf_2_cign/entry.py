from tf_2_cign.cign import Cign
from tf_2_cign.data.fashion_mnist import FashionMnist
from tf_2_cign.fashion_net.fashion_cign import FashionCign

# Hyper-parameters
input_dims = (28, 28, 3)
degree_list = []
batch_size = 128
epoch_count = 100
decision_drop_probability = 0.0
drop_probability = 0.0
classification_wd = 0.0
decision_wd = 0.0

# FashionNet parameters
filter_counts = [32, 32, 32]
kernel_sizes = [5, 5, 1]
hidden_layers = [128, 64]
decision_dimensions = [128, 128]
node_build_funcs = [FashionCign.root_func]

if __name__ == "__main__":
    fashion_mnist = FashionMnist(batch_size=batch_size)
    cign = FashionCign(input_dims=input_dims,
                       node_degrees=degree_list,
                       filter_counts=filter_counts,
                       kernel_sizes=kernel_sizes,
                       hidden_layers=kernel_sizes,
                       decision_drop_probability=decision_drop_probability,
                       classification_drop_probability=drop_probability,
                       decision_wd=decision_wd,
                       classification_wd=classification_wd,
                       decision_dimensions=decision_dimensions,
                       node_build_funcs=node_build_funcs)
    cign.build_network()





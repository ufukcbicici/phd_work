import simple_tf.cifar_nets.cifar_entry as cifar_entry
#
# # print("Deneme deneme")
# # if __name__ == "__main__":
# #     # main()
# #     print("Deneme deneme")
#     # cifar_entry.cifar100_training()
#
# import tensorflow as tf
#
#
# print(tf.__version__)

import tensorflow as tf

from algorithms.threshold_optimization_algorithms import threshold_optimization_entry
from simple_tf.cigj import cigj_full_training_entry, cigj_approx_training_entry
from simple_tf.fashion_net import fashion_cign_entry
from tf_experiments.multi_gpu_experiments import multi_gpu

if __name__ == "__main__":
    print("Main - Hello World. With Import.")
    print(tf.__version__)
    # cifar_entry.cifar100_training()
    # cigj_full_training_entry.cigj_training()
    # multi_gpu.experiment()
    # multi_gpu.experiment_with_towers()
    # multi_gpu.experiment_with_custom_batch_norms()
    # cifar_entry.cifar100_multi_gpu_training()
    cifar_entry.cifar_100_training()
    # fashion_cign_entry.fashion_net_training()
    # threshold_optimization_entry.main()




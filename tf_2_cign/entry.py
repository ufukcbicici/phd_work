from tf_2_cign.cign import Cign

# Hyper-parameters
input_dims = (28, 28, 3)
degree_list = [2, 2]

if __name__ == "__main__":
    cign = Cign(input_dims=input_dims, node_degrees=degree_list)
    cign.build_network()

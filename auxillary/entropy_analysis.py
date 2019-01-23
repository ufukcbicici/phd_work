import numpy as np
from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger
from auxillary.general_utility_funcs import UtilityFuncs


def analyze_entropy(run_id, dataset_type):
    query = "SELECT * FROM run_kv_store "
    condition = "WHERE RunId={0} AND ".format(run_id)
    condition += "Key LIKE \"%{0}%\" AND ".format(dataset_type)
    condition += "Key LIKE \"%True Label%\"".format(dataset_type)
    query += condition
    rows = DbLogger.read_query(query=query)
    iteration_dict = {}
    for row in rows:
        iteration = row[1]
        key_value = row[2]
        value = row[3]
        if iteration not in iteration_dict:
            iteration_dict[iteration] = {}
        i0 = key_value.index("Leaf:")
        i1 = key_value.index("True Label:")
        leaf_id = key_value[i0 + len("Leaf:"):i1-1]
        label_id = key_value[i1 + 1 + len("True Label"):len(key_value)]
        if leaf_id not in iteration_dict[iteration]:
            iteration_dict[iteration][leaf_id] = {}
        iteration_dict[iteration][leaf_id][label_id] = value
    # Analyze entropies
    for iteration in iteration_dict.keys():
        leaf_dict = iteration_dict[iteration]
        for leaf_id, freq_dict in leaf_dict.items():
            freq_array = np.zeros(shape=(len(freq_dict)))
            for label_id, freq in freq_dict.items():
                freq_array[int(label_id)] = freq
                if freq == 0:
                    print("Zero")
            prob_distribution = freq_array / np.sum(freq_array)
            entropy = UtilityFuncs.calculate_distribution_entropy(distribution=prob_distribution)
    print("X")


analyze_entropy(run_id=3, dataset_type=DatasetTypes.test)

from auxillary.constants import DatasetTypes
from auxillary.db_logger import DbLogger


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
        label_id = key_value[i1 + len("True Label"):len(key_value)]
        if leaf_id not in iteration_dict[iteration]:
            iteration_dict[iteration][leaf_id] = {}
        iteration_dict[iteration][leaf_id][label_id] = value
        print("X")


analyze_entropy(run_id=3, dataset_type=DatasetTypes.test)

import os
from distutils.dir_util import copy_tree

# Read results
from simple_tf.global_params import GlobalConstants
from zipfile import ZipFile

base_path = os.path.join("/cta", "users", "ucbicici", "phd_work")
# base_path = os.path.join("D:/", "phd_work")
results_file_path = os.path.join(base_path, "algorithms", "dgx_results.txt")
curr_path = os.path.join(base_path, "algorithms")
source_path = os.path.join(curr_path, "..", GlobalConstants.MODEL_SAVE_FOLDER)
destination_path = os.path.join(curr_path, "..", "important_results")
selected_files_count = 5


def move_files():
    results_file = open(results_file_path, 'r')
    lines = results_file.readlines()
    count = 0
    # Strips the newline character
    results_arr = []
    for line in lines:
        parts = line.split(sep="|")
        results_arr.append((parts[0], parts[3]))

    results_sorted = sorted(results_arr, key=lambda tpl: tpl[1], reverse=True)
    selected_results = results_sorted[:selected_files_count]
    selected_ids = [tpl[0] for tpl in selected_results]
    selected_ids.append("451")

    folder_list = os.listdir(source_path)
    selected_folders = []
    for folder_name in folder_list:
        includes = any(["_{0}_".format(id_) in folder_name for id_ in selected_ids])
        if includes:
            selected_folders.append(folder_name)
            selected_folder_path = os.path.join(source_path, folder_name)
            destination_folder_path = os.path.join(destination_path, folder_name)
            print("Copying {0}".format(folder_name))
            copy_tree(selected_folder_path, destination_folder_path)

    # Compress all files
    # with ZipFile(os.path.join(base_path, "important_results.zip"), mode='w') as zf:
    #     zf.write(destination_path)
    # print("X")


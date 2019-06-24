from auxillary.path_query import execute_path_query
import sys

print('Number of arguments:', len(sys.argv), 'arguments.')

# min_id = sys.argv[1]
# max_id = sys.argv[2]

min_id = 26
max_id = 29


execute_path_query(min_id=min_id, max_id=max_id)
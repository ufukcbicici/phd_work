import sys

from auxillary.db_logger import DbLogger


def get_query(min_run_id, max_run_id, condition, iteration_lower_limit, add_union=True):
    qry = "SELECT RunId, Threshold, IsWeighted, Avg(Accuracy) AS Accuracy, AVG(LeafEvaluated) AS LeafEvaluated, COUNT(1) AS CNT\n"
    qry += "FROM\n"
    qry += "(\n"
    qry += "SELECT RunId,Max(RunId),Min(RunId),Threshold, IsWeighted, Avg(Accuracy) AS Accuracy, " \
           "AVG(LeafEvaluated) AS LeafEvaluated,COUNT(1) AS CNT\n"
    qry += "FROM {0} WHERE Iteration >= {1}\n".format(DbLogger.multipath_results_table,
                                                      iteration_lower_limit)
    qry += "AND RunId >= {0} AND RunId <= {1}\n".format(min_run_id, max_run_id)
    qry += "GROUP BY Threshold, IsWeighted\n"
    qry += ")\n"
    qry += condition
    qry += "\n"
    qry += "GROUP BY IsWeighted\n"
    if add_union:
        qry += "UNION ALL\n"
    return qry


print('Number of arguments:', len(sys.argv), 'arguments.')

min_id = sys.argv[1]
max_id = sys.argv[2]


def execute_path_query():
    step1 = 250
    step2 = 500
    low_limit = 10000
    mid_limit = 11000
    max_limit = 35000
    intervals = [(i, i + step1) for i in range(low_limit, mid_limit, step1)]
    intervals.extend([(i, i + step2) for i in range(mid_limit, max_limit, step2)])
    iteration_lower_limit = 43201
    query = get_query(min_run_id=min_id, max_run_id=max_id, condition="WHERE LeafEvaluated = {0}".format(low_limit),
                      iteration_lower_limit=iteration_lower_limit)
    for interval in intervals:
        query += get_query(min_run_id=min_id, max_run_id=max_id,
                           condition="WHERE {0} < LeafEvaluated AND LeafEvaluated <= {1}".format(interval[0],
                                                                                                 interval[1]),
                           iteration_lower_limit=iteration_lower_limit)
    query += get_query(min_run_id=min_id, max_run_id=max_id, condition="WHERE {0} < LeafEvaluated".format(max_limit),
                       iteration_lower_limit=iteration_lower_limit, add_union=False)
    print(query)

    rows = DbLogger.read_query(query=query)
    for row in rows:
        print(row)

# SELECT RunId, Avg(Accuracy) AS Accuracy, AVG(LeafEvaluated) AS LeafEvaluated, COUNT(1) AS CNT
# FROM
# (
# 	SELECT RunId,Max(RunId),Min(RunId),Threshold, IsWeighted, Avg(Accuracy) AS Accuracy, AVG(LeafEvaluated) AS LeafEvaluated,COUNT(1) AS CNT
# 	FROM multipath_results WHERE Iteration >= 43201 And IsWeighted = 0
# 	AND RunId >= 1764 AND RunId <= 1793
# 	GROUP BY Threshold, IsWeighted
# 	ORDER BY Accuracy DESC
# )
# WHERE 15000 < LeafEvaluated AND LeafEvaluated <= 20000

SELECT * FROM multipath_results 

SELECT Max(RunId) FROM run_meta_data --777

SELECT * FROM multipath_results WHERE RunID = 777 ORDER BY Threshold
SELECT * FROM multipath_results_v2 WHERE RunID = 777 ORDER BY Threshold

SELECT multipath_results.RunID, multipath_results.Iteration, multipath_results.IsWeighted, multipath_results.Threshold,
multipath_results.Accuracy - multipath_results_v2.Accuracy AS AccuracyDiff,
multipath_results.LeafEvaluated - multipath_results_v2.LeafEvaluated AS LeafEvaluatedDiff
FROM multipath_results LEFT JOIN multipath_results_v2 
ON 
multipath_results.RunID = multipath_results_v2.RunID 
AND multipath_results.Iteration = multipath_results_v2.Iteration
AND multipath_results.IsWeighted = multipath_results_v2.IsWeighted
AND multipath_results.Threshold = multipath_results_v2.Threshold
WHERE multipath_results.RunID = 777

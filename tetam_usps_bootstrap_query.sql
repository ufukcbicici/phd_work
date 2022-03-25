SELECT
RunId,
Max(RunId),
Min(RunId),
Max(ValidationAccuracy) AS MaxValidationAccuracy,
Min(ValidationAccuracy) AS MinValidationAccuracy,
Avg(ValidationAccuracy) AS AvgValidationAccuracy,
Avg(TestAccuracy) AS AvgCorrectedValidationAccuracy,
Avg(TrainingAccuracy) AS AvgTrainingAccuracy,
COUNT(1) AS CNT,
SUBSTR(Explanation, INSTR(Explanation,"Wd:") + LENGTH("Wd:"), 10) AS Wd,
SUBSTR(Explanation, INSTR(Explanation,"Iteration:0 Value:") + LENGTH("Iteration:0 Value:"), 10) AS Lr,
SUBSTR(Explanation, INSTR(Explanation,"Info Gain Balance Coefficient:") + LENGTH("Info Gain Balance Coefficient:"), 10) AS IG_Balance
FROM
(
SELECT logs_table.*,run_meta_data.Explanation FROM logs_table
LEFT JOIN run_meta_data ON logs_table.RunId = run_meta_data.RunId
WHERE logs_table.Epoch >= 185
AND
Explanation LIKE "%Batch Size:125%"
AND
Explanation LIKE "%Bootstrap%"
)
GROUP BY Explanation
ORDER BY AvgValidationAccuracy DESC


--dblogger2.db
SELECT COUNT(*) FROM run_meta_data WHERE Explanation LIKE "%Bootstrap%"
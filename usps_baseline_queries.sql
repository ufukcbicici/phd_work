SELECT Max(RunID) FROM run_meta_data

SELECT Explanation FROM run_meta_data WHERE RunID = 2220


--Thick Baseline
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
Explanation LIKE "%USPS - Baseline - Thick Baseline%"
AND
Explanation LIKE "%Param Count:141130%"
)
GROUP BY Explanation
ORDER BY AvgValidationAccuracy DESC


SELECT Explanation FROM run_meta_data WHERE RunID = 2200

--Thin
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
Explanation LIKE "%USPS - Baseline - Thin Baseline%"
AND
Explanation LIKE "%Param Count:9586%"
)
GROUP BY Explanation
ORDER BY AvgValidationAccuracy DESC

SELECT Explanation FROM run_meta_data WHERE RunID = 2303

--Random
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
Explanation LIKE "%USPS - CIGN - Random Sample%"
AND
Explanation LIKE "%[32, 24, 16]%"
)
GROUP BY Explanation
ORDER BY AvgValidationAccuracy DESC

--All Paths
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
Explanation LIKE "%USPS - CIGN - All Samples Routed%"
AND
Explanation LIKE "%[32, 24, 16]%"
)
GROUP BY Explanation
ORDER BY AvgValidationAccuracy DESC
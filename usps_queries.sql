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
SUBSTR(Explanation, INSTR(Explanation,"Classification Dropout Probability:") + LENGTH("Classification Dropout Probability:"), 4) AS ClassificationDropout,
SUBSTR(Explanation, INSTR(Explanation,"Decision Dropout Probability:") + LENGTH("Decision Dropout Probability:"), 4) AS DecisionDropout
FROM
(
	SELECT logs_table.*,run_meta_data.Explanation FROM logs_table
	LEFT JOIN run_meta_data ON logs_table.RunId = run_meta_data.RunId
	WHERE logs_table.Epoch >= 190
	AND
	Explanation LIKE "%USPS - Baseline - Bayesian Optimization - ReLU Fixed%" AND
	Explanation LIKE "%Param Count:19226%"
)
GROUP BY Explanation
ORDER BY AvgValidationAccuracy DESC


SELECT * FROM run_meta_data WHERE RunID = 1372 --1375, 1374
********Lr Settings********
Iteration:0 Value:0.09746176130054329
Iteration:2500 Value:0.048730880650271646
Iteration:5000 Value:0.024365440325135823
Iteration:7500 Value:0.0024365440325135822
********Lr Settings********
Wd:0.0008120373740661416

SELECT * FROM run_meta_data WHERE RunID = 1375 
********Lr Settings********
Iteration:0 Value:0.09767504912890895
Iteration:2500 Value:0.048837524564454476
Iteration:5000 Value:0.024418762282227238
Iteration:7500 Value:0.0024418762282227236
********Lr Settings********
Wd:0.0009627508498408631

SELECT * FROM run_meta_data WHERE RunID = 1374 
********Lr Settings********
Iteration:0 Value:0.09745033033453893
Iteration:2500 Value:0.04872516516726946
Iteration:5000 Value:0.02436258258363473
Iteration:7500 Value:0.002436258258363473
********Lr Settings********
Wd:0.0008218704639722474

USPS - Baseline - Bayesian Optimization - ReLU Fixed - Thin Baseline Best Hyperparameters

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
SUBSTR(Explanation, INSTR(Explanation,"Classification Dropout Probability:") + LENGTH("Classification Dropout Probability:"), 4) AS ClassificationDropout,
SUBSTR(Explanation, INSTR(Explanation,"Decision Dropout Probability:") + LENGTH("Decision Dropout Probability:"), 4) AS DecisionDropout
FROM
(
	SELECT logs_table.*,run_meta_data.Explanation FROM logs_table
	LEFT JOIN run_meta_data ON logs_table.RunId = run_meta_data.RunId
	WHERE logs_table.Epoch >= 190
	AND
	Explanation LIKE "%USPS - Baseline - Bayesian Optimization - ReLU Fixed - Thin Baseline Best Hyperparameters%" AND
	Explanation LIKE "%Param Count:19226%"
)
GROUP BY Explanation
ORDER BY AvgValidationAccuracy DESC


SELECT Max(RunID) FROM run_meta_data
SELECT * FROM run_meta_data WHERE RunID = 1466

SELECT * FROM logs_table WHERE RunID = 1454


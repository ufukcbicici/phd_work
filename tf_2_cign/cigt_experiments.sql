--Experiments: Network Name:Lenet CIGT - Dropout Optimization-1
--Started at 10/4/2022
--Started on: TETAM - cigt_logger.db

SELECT AVG(TrainingAccuracy) AS TrainingAccuracy,
       AVG(TestAccuracy) AS TestAccuracy,
       run_parameters.Value,
       COUNT(1) AS CNT
FROM logs_table LEFT JOIN run_parameters ON logs_table.RunID = run_parameters.RunID
WHERE logs_table.RunID IN
(SELECT logs_table.RunId FROM logs_table LEFT JOIN run_meta_data ON
logs_table.RunId = run_meta_data.RunId
WHERE run_meta_data.Explanation LIKE "%Lenet CIGT - Dropout Optimization-1%") AND
run_parameters.Parameter = "Classification Dropout" AND logs_table.Epoch >= 115
GROUP BY run_parameters.Value

SELECT AVG(TrainingAccuracy) AS TrainingAccuracy,
       AVG(TestAccuracy) AS TestAccuracy,
       A.Value AS ClassificationDropout,
       B.Value AS InformationGainBalanceCoefficient,
       C.Value AS DecisionLossCoefficient,
       COUNT(1) AS CNT
FROM logs_table
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Classification Dropout") AS A ON
    logs_table.RunID = A.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Information Gain Balance Coefficient") AS B ON
    logs_table.RunID = B.RunID
    LEFT JOIN (SELECT * FROM run_parameters WHERE run_parameters.Parameter = "Decision Loss Coeff") AS C ON
    logs_table.RunID = C.RunID
WHERE logs_table.RunID IN
(SELECT logs_table.RunId FROM logs_table LEFT JOIN run_meta_data ON
logs_table.RunId = run_meta_data.RunId
WHERE run_meta_data.Explanation LIKE "%Lenet CIGT - Dropout Optimization-1%") AND logs_table.Epoch >= 115
GROUP BY ClassificationDropout, InformationGainBalanceCoefficient, DecisionLossCoefficient

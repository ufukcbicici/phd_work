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

SELECT AVG(TrainingAccuracy) AS TrainingAccuracy,
       AVG(TestAccuracy) AS TestAccuracy,
       MIN(A.Value) AS ClassificationDropoutMin,
       MAX(A.Value) AS ClassificationDropoutMax,
       MIN(B.Value) AS InformationGainBalanceCoefficientMin,
       MAX(B.Value) AS InformationGainBalanceCoefficientMax,
       MIN(C.Value) AS DecisionLossCoefficientMin,
       MAX(C.Value) AS DecisionLossCoefficientMax,
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
WHERE run_meta_data.Explanation LIKE "%Lenet CIGT - Dropout Optimization-1%") AND logs_table.Epoch >= 115 AND
      A.Value IN ("0.35", "0.4")
GROUP BY logs_table.RunId
ORDER BY TestAccuracy DESC

SELECT logs_table.RunId,
       AVG(TrainingAccuracy) AS TrainingAccuracy,
       AVG(TestAccuracy) AS TestAccuracy,
       MIN(A.Value) AS ClassificationDropoutMin,
       MAX(A.Value) AS ClassificationDropoutMax,
       MIN(B.Value) AS InformationGainBalanceCoefficientMin,
       MAX(B.Value) AS InformationGainBalanceCoefficientMax,
       MIN(C.Value) AS DecisionLossCoefficientMin,
       MAX(C.Value) AS DecisionLossCoefficientMax,
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
WHERE run_meta_data.Explanation LIKE "%Lenet CIGT - Dropout Optimization-1%") AND logs_table.Epoch >= 115 AND
      A.Value = "0.35"
GROUP BY logs_table.RunId

--GROUP BY ClassificationDropout, InformationGainBalanceCoefficient, DecisionLossCoefficient

--Check after experiment 91 for Dropout=0.05


--Experiments: Network Name:Lenet CIGT - IG Balance Optimization-1
--Started at 15/4/2022
--Started on: TETAM - cigt_logger.db
--Starts with RunId 296

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
WHERE run_meta_data.Explanation LIKE "%Lenet CIGT - IG Balance Optimization-1%") AND logs_table.Epoch >= 115
GROUP BY ClassificationDropout, InformationGainBalanceCoefficient, DecisionLossCoefficient


SELECT logs_table.RunId,
       AVG(TrainingAccuracy) AS TrainingAccuracy,
       AVG(TestAccuracy) AS TestAccuracy,
       MIN(A.Value) AS ClassificationDropoutMin,
       MAX(A.Value) AS ClassificationDropoutMax,
       MIN(B.Value) AS InformationGainBalanceCoefficientMin,
       MAX(B.Value) AS InformationGainBalanceCoefficientMax,
       MIN(C.Value) AS DecisionLossCoefficientMin,
       MAX(C.Value) AS DecisionLossCoefficientMax,
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
WHERE run_meta_data.Explanation LIKE "%Lenet CIGT - IG Balance Optimization-1%") AND logs_table.Epoch >= 115
GROUP BY logs_table.RunId

SELECT logs_table.RunId,
       AVG(TrainingAccuracy) AS TrainingAccuracy,
       AVG(TestAccuracy) AS TestAccuracy,
       MIN(A.Value) AS ClassificationDropoutMin,
       MAX(A.Value) AS ClassificationDropoutMax,
       MIN(B.Value) AS InformationGainBalanceCoefficientMin,
       MAX(B.Value) AS InformationGainBalanceCoefficientMax,
       MIN(C.Value) AS DecisionLossCoefficientMin,
       MAX(C.Value) AS DecisionLossCoefficientMax,
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
WHERE run_meta_data.Explanation LIKE "%Lenet CIGT - IG Balance Optimization-1%") AND logs_table.Epoch >= 115 AND
      A.Value IN ("0.35") AND B.Value IN ("3.0")
GROUP BY logs_table.RunId
ORDER BY TestAccuracy DESC


--Experiments: Network Name:Lenet CIGT - Bayesian Optimization
--Started at 18/4/2022
--Started on: Home Lab - dblogger.db

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
WHERE run_meta_data.Explanation LIKE "%Lenet CIGT - Bayesian Optimization%") AND logs_table.Epoch >= 115
GROUP BY ClassificationDropout, InformationGainBalanceCoefficient, DecisionLossCoefficient
ORDER BY TestAccuracy DESC


--Experiments: Network Name:Lenet CIGT - Bayesian Optimization - [2,2]- [32,64,64] - [256,128]
--Started at 19/4/2022
--Started on: Tetam Tuna - cigt_logger.db
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
WHERE run_meta_data.Explanation
          LIKE "%Lenet CIGT - Bayesian Optimization - [2,2]- [32,64,64] - [256,128]%") AND logs_table.Epoch >= 115
GROUP BY ClassificationDropout, InformationGainBalanceCoefficient, DecisionLossCoefficient
ORDER BY TestAccuracy DESC


--Experiments: Network Name:Network Name:Lenet CIGT - Decision Loss Coeff Optimization-1
--Started at 21/4/2022
--Started on: TETAM - cigt_logger.db

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
WHERE run_meta_data.Explanation LIKE "%Lenet CIGT - Decision Loss Coeff Optimization-1%") AND logs_table.Epoch >= 115
GROUP BY ClassificationDropout, InformationGainBalanceCoefficient, DecisionLossCoefficient
SELECT * FROM dataset_link

SELECT * FROM threshold_optimization WHERE NetworkName = "USPS_CIGN" AND RunID = 1596

SELECT COUNT(*) FROM threshold_optimization WHERE NetworkName = "USPS_CIGN" AND RunID = 1596



SELECT Iterations, Lambda, Xi, AVG(ValAccuracy) AS ValAccuracy, AVG(ValComputationOverload) AS ValComputationOverload, 
AVG(TestScore) AS TestScore, AVG(TestAccuracy) AS TestAccuracy, AVG(TestComputationOverload) AS TestComputationOverload, COUNT(1) AS CNT,
0.5*AVG(ValAccuracy) + 0.5*AVG(TestAccuracy) AS MeanAccuracy,
0.5*AVG(ValComputationOverload) + 0.5*AVG(TestComputationOverload) AS MeanComputationOverload
FROM
(
	SELECT Iterations, Lambda, Xi, TimeStamp, Max(ValScore) AS MaxValScore, ValAccuracy, ValComputationOverload, TestScore, TestAccuracy, TestComputationOverload FROM
		threshold_optimization WHERE RunID IN (1596) GROUP BY TimeStamp
)
GROUP BY Xi, Lambda
ORDER BY MeanAccuracy DESC

SELECT Iterations, Lambda, Xi, 
AVG(ValAccuracy) AS ValAccuracy, 
AVG(ValComputationOverload) AS ValComputationOverload, 
AVG(ValF1Micro) AS ValF1Micro,
AVG(ValF1Macro) AS ValF1Macro,
AVG(TestScore) AS TestScore, 
AVG(TestAccuracy) AS TestAccuracy, 
AVG(TestComputationOverload) AS TestComputationOverload, 
AVG(TestF1Micro) AS TestF1Micro,
AVG(TestF1Macro) AS TestF1Macro,
COUNT(1) AS CNT,
0.5*AVG(ValAccuracy) + 0.5*AVG(TestAccuracy) AS MeanAccuracy,
0.5*AVG(ValComputationOverload) + 0.5*AVG(TestComputationOverload) AS MeanComputationOverload
FROM
(
	SELECT Iterations, Lambda, Xi, TimeStamp, Max(ValScore) AS MaxValScore, ValF1Micro, ValF1Macro,
	ValAccuracy, ValComputationOverload, TestScore, TestF1Micro, TestF1Macro, TestAccuracy, TestComputationOverload FROM
		threshold_optimization_f1_metrics WHERE RunID IN (1577) GROUP BY TimeStamp
)
GROUP BY Xi, Lambda
ORDER BY MeanAccuracy DESC

SELECT Xi, Lambda, COUNT(1) AS CNT FROM threshold_optimization WHERE RunID IN (1577) GROUP BY Xi, Lambda


SELECT Iterations, Lambda, Xi, AVG(ValAccuracy) AS ValAccuracy, AVG(ValComputationOverload) AS ValComputationOverload, 
AVG(TestScore) AS TestScore, AVG(TestAccuracy) AS TestAccuracy, AVG(TestComputationOverload) AS TestComputationOverload, COUNT(1) AS CNT,
0.5*AVG(ValAccuracy) + 0.5*AVG(TestAccuracy) AS MeanAccuracy,
0.5*AVG(ValComputationOverload) + 0.5*AVG(TestComputationOverload) AS MeanComputationOverload
FROM
(
	SELECT Iterations, Lambda, Xi, TimeStamp, Max(ValScore) AS MaxValScore, ValAccuracy, ValComputationOverload, TestScore, TestAccuracy, TestComputationOverload FROM
		threshold_optimization WHERE RunID IN (592, 594, 618, 679, 717) GROUP BY TimeStamp
)
GROUP BY Xi, Lambda
ORDER BY MeanAccuracy DESC

SELECT * FROM threshold_optimization_f1_metrics


CREATE TABLE "threshold_optimization_f1_metrics" 
( 
`RunId` INTEGER, 
`NetworkName` TEXT, 
`Iterations` TEXT, 
`Lambda` NUMERIC, 
`Method` INTEGER, 
`ValScore` NUMERIC, 
`ValAccuracy` NUMERIC, 
`ValComputationOverload` NUMERIC, 
`ValF1Micro` NUMERIC, 
`ValF1Macro` NUMERIC, 
`TestScore` NUMERIC, 
`TestAccuracy` NUMERIC, 
`TestComputationOverload` NUMERIC, 
`TestF1Micro` NUMERIC, 
`TestF1Macro` NUMERIC, 
`MethodName` TEXT, 
`xi` NUMERIC, 
`Timestamp` TEXT 
)



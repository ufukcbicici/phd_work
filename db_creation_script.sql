BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "run_results" (
	"RunId"	INTEGER,
	"Explanation"	TEXT,
	"TestAccuracy"	REAL,
	"Type"	TEXT
);
CREATE TABLE IF NOT EXISTS "run_meta_data" (
	"RunId"	INTEGER,
	"Explanation"	TEXT
);
CREATE TABLE IF NOT EXISTS "run_kv_store" (
	"RunId"	INTEGER,
	"Iteration"	INTEGER,
	"Key"	TEXT,
	"Value"	REAL
);
CREATE TABLE IF NOT EXISTS "multipath_results" (
	"RunId"	INTEGER,
	"Iteration"	INTEGER,
	"IsWeighted"	INTEGER,
	"Threshold"	REAL,
	"Accuracy"	REAL,
	"LeafEvaluated"	INTEGER
);
CREATE TABLE IF NOT EXISTS "leaf_info_table" (
	"LeafIndex"	INTEGER,
	"NumOfSamples"	INTEGER,
	"Iteration"	INTEGER,
	"RunId"	INTEGER
);
CREATE TABLE IF NOT EXISTS "info_gain_optimization" (
	"GradRatio"	NUMERIC,
	"LogRatio"	NUMERIC,
	"GradType"	INTEGER,
	"InfoGain"	NUMERIC,
	"Iteration"	INTEGER,
	"lr"	NUMERIC,
	"RunId"	INTEGER,
	"Model"	TEXT,
	"SampleCount"	INTEGER
);
CREATE TABLE IF NOT EXISTS "info_gain_derivatives" (
	"GradRatio"	NUMERIC,
	"LogRatio"	NUMERIC,
	"GradType"	INTEGER,
	"AvgBias"	NUMERIC,
	"AvgVariance"	NUMERIC,
	"ArgName"	TEXT,
	"SampleCount"	INTEGER,
	"AvgAbsoluteBias"	NUMERIC,
	"ModelName"	TEXT
);
CREATE TABLE IF NOT EXISTS "confusion_matrices" (
	"RunId"	INTEGER,
	"Dataset"	INTEGER,
	"LeafIndex"	INTEGER,
	"Iteration"	INTEGER,
	"TrueLabel"	INTEGER,
	"PredictedLabel"	INTEGER,
	"Frequency"	INTEGER
);
CREATE TABLE IF NOT EXISTS "multipath_results_v2" (
	"RunId"	INTEGER,
	"Iteration"	INTEGER,
	"IsWeighted"	INTEGER,
	"Threshold"	REAL,
	"Accuracy"	REAL,
	"LeafEvaluated"	INTEGER
);
CREATE TABLE IF NOT EXISTS "multipath_regression" (
	"RunId"	NUMERIC,
	"Iteration"	INTEGER,
	"ValMse"	NUMERIC,
	"ValAccuracySimpleAvg"	NUMERIC,
	"ValAccuracyEnsembleAvg"	NUMERIC,
	"ValLoss"	NUMERIC,
	"TestMse"	NUMERIC,
	"TestAccuracySimpleAvg"	NUMERIC,
	"TestAccuracyEnsembleAvg"	NUMERIC,
	"TestLoss"	NUMERIC,
	"L2"	NUMERIC
);
CREATE TABLE IF NOT EXISTS "multipath_classification" (
	"RunId"	NUMERIC,
	"Iteration"	INTEGER,
	"ValAccuracy"	NUMERIC,
	"ValAccuracySimpleAvg"	NUMERIC,
	"ValAccuracyEnsembleAvg"	NUMERIC,
	"ValLoss"	NUMERIC,
	"TestAccuracy"	NUMERIC,
	"TestAccuracySimpleAvg"	NUMERIC,
	"TestAccuracyEnsembleAvg"	NUMERIC,
	"TestLoss"	NUMERIC,
	"L2"	NUMERIC
);
CREATE TABLE IF NOT EXISTS "threshold_optimization_old" (
	"RunId"	INTEGER,
	"NetworkName"	TEXT,
	"Iterations"	TEXT,
	"Lambda"	NUMERIC,
	"Method"	INTEGER,
	"ValScore"	NUMERIC,
	"ValAccuracy"	NUMERIC,
	"ValComputationOverload"	NUMERIC,
	"TestScore"	NUMERIC,
	"TestAccuracy"	NUMERIC,
	"TestComputationOverload"	NUMERIC,
	"MethodName"	TEXT,
	"xi"	NUMERIC,
	"Timestamp"	TEXT
);
CREATE TABLE IF NOT EXISTS "threshold_optimization" (
	"RunId"	INTEGER,
	"NetworkName"	TEXT,
	"Iterations"	TEXT,
	"Lambda"	NUMERIC,
	"Method"	INTEGER,
	"ValScore"	NUMERIC,
	"ValAccuracy"	NUMERIC,
	"ValComputationOverload"	NUMERIC,
	"TestScore"	NUMERIC,
	"TestAccuracy"	NUMERIC,
	"TestComputationOverload"	NUMERIC,
	"MethodName"	TEXT,
	"xi"	NUMERIC,
	"Timestamp"	TEXT
);
CREATE TABLE IF NOT EXISTS "policy_gradients_results" (
	"RunId"	INTEGER,
	"Iteration"	INTEGER,
	"ValidationPolicyValue"	NUMERIC,
	"TestPolicyValue"	NUMERIC,
	"IsTestData"	NUMERIC,
	"IgnoreInvalidActions"	NUMERIC,
	"CombineWithIg"	NUMERIC,
	"Accuracy"	NUMERIC,
	"ComputationOverload"	NUMERIC
);
CREATE TABLE IF NOT EXISTS "policy_gradients_parameters" (
	"RunId"	INTEGER,
	"FoldId"	INTEGER,
	"NetworkRunId"	INTEGER,
	"NetworkName"	TEXT,
	"StateSampleCount"	INTEGER,
	"TrajectoryPerSampleCount"	INTEGER,
	"LambdaMacCost"	NUMERIC,
	"l2Lambda"	NUMERIC,
	"Val_Ml_Accuracy"	NUMERIC,
	"Test_Ml_Accuracy"	NUMERIC
);
CREATE TABLE IF NOT EXISTS "dataset_link" (
	"NetworkName"	TEXT,
	"NetworkId"	INTEGER,
	"Iteration"	INTEGER,
	"FeatureName"	TEXT,
	"NodeId"	INTEGER,
	"SampleId"	INTEGER,
	"SampleIdForIteration"	INTEGER
);
CREATE TABLE IF NOT EXISTS "deep_q_learning_logs" (
	"RunId"	INTEGER,
	"Iteration"	INTEGER,
	"training_mean_policy_value"	NUMERIC,
	"training_mse"	NUMERIC,
	"training_accuracy"	NUMERIC,
	"training_computation_cost"	NUMERIC,
	"test_mean_policy_value"	NUMERIC,
	"test_mse"	NUMERIC,
	"test_accuracy"	NUMERIC,
	"test_computation_cost"	NUMERIC
);
CREATE TABLE IF NOT EXISTS "threshold_optimization_f1_metrics" (
	"RunId"	TEXT,
	"NetworkName"	TEXT,
	"Iterations"	TEXT,
	"Lambda"	NUMERIC,
	"Method"	INTEGER,
	"ValScore"	NUMERIC,
	"ValAccuracy"	NUMERIC,
	"ValComputationOverload"	NUMERIC,
	"ValF1Micro"	NUMERIC,
	"ValF1Macro"	NUMERIC,
	"TestScore"	NUMERIC,
	"TestAccuracy"	NUMERIC,
	"TestComputationOverload"	NUMERIC,
	"TestF1Micro"	NUMERIC,
	"TestF1Macro"	NUMERIC,
	"MethodName"	TEXT,
	"xi"	NUMERIC,
	"Timestamp"	TEXT
);
CREATE TABLE IF NOT EXISTS "q_net_anomaly_logs" (
	"RunId"	INTEGER,
	"Epoch"	INTEGER,
	"LevelId"	INTEGER,
	"DataType"	TEXT,
	"ClassificationReport"	TEXT,
	"Auc"	NUMERIC
);
CREATE TABLE IF NOT EXISTS "logs_table" (
	"RunId"	INTEGER,
	"Iteration"	INTEGER,
	"Epoch"	INTEGER,
	"TrainingAccuracy"	NUMERIC,
	"ValidationAccuracy"	NUMERIC,
	"TestAccuracy"	NUMERIC,
	"MeanTime"	NUMERIC,
	"TrainingError"	NUMERIC,
	"ValidationError"	REAL,
	"BnnName"	TEXT
);
CREATE TABLE IF NOT EXISTS "run_parameters" (
	"RunId"	INTEGER,
	"Parameter"	TEXT,
	"Value"	TEXT
);
COMMIT;

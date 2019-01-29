import numpy as np
import pandas as pd


class CategoricalEncoder:
    def __init__(self):
        self.categoryMap = {}

    def fit(self, df, categorical_columns):
        for categorical_column in categorical_columns:
            # Unique values
            unique_values = set(df[categorical_column].unique().tolist())
            val_dict = {}
            counter = 0
            # Check if we have "Empty". If so, it will be all zeros.
            if "Empty" in unique_values:
                val_dict["Empty"] = counter
                counter += 1
            for val in unique_values:
                if val not in val_dict:
                    val_dict[val] = counter
                    counter += 1
            assert unique_values.issubset(set(val_dict.keys())) and set(val_dict.keys()).issubset(unique_values)
            self.categoryMap[categorical_column] = val_dict

    def transform(self, df):
        for categorical_column, val_map in self.categoryMap.items():
            assert categorical_column in self.categoryMap
            original_column = df[categorical_column].copy()
            # Drop the categorical column
            df = df.drop([categorical_column], axis=1)
            encoded_data = np.zeros(shape=(df.shape[0], len(val_map) - 1))
            for row_index, val in original_column.iteritems():
                encoded = val_map[val]
                if encoded == 0:
                    continue
                encoded_data[row_index, encoded-1] = 1.0
            df = pd.concat([df, pd.DataFrame(encoded_data)], axis=1)
            # # Add onehot columns
            # for i in range(len(val_map)):
            #     df["{0}_{1}".format(categorical_column, i + 1)] = 0.0
            # # Encode from the original column
            # for row_index, val in original_column.iteritems():
            #     encoded = val_map[val]
            #     if encoded == 0:
            #         continue
            #     col_name = "{0}_{1}".format(categorical_column, encoded)
            #     col_index = df.columns.get_loc(col_name)
            #     df.iloc[row_index, col_index] = 1.0
            print("X")
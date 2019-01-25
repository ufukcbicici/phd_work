import numpy as np
import pandas as pd


train_data = pd.read_csv("C://Users//t67rt//Desktop//phd_work//phd_work//geophy//data//train.csv")
test_data = pd.read_csv("C://Users//t67rt//Desktop//phd_work//phd_work//geophy//data//test.csv")
categorical_columns = ["borough", "schooldist", "council", "zipcode", "firecomp", "policeprct",
                       "healthcenterdistrict", "healtharea", "sanitboro", "sanitdistrict", "sanitsub",
                       "zonedist1", "zonedist2", "zonedist3", "zonedist4", "overlay1", "overlay2", "spdist1",
                       "spdist2", "spdist3", "ltdheight", "splitzone", "landuse", "ext", "proxcode",
                       "irrlotcode", "lottype", "bsmtcode", "yearbuilt", "yearalter1", "yearalter2", "histdist",
                       "landmark", "tract2010", "zonemap"]
numerical_columns = ["easements", "block", "lot", "lotarea", "bldgarea", "comarea", "resarea", "officearea", "officearea",
                     "retailarea", "garagearea", "strgearea", "factryarea", "otherarea", "numbldgs", "numfloors",
                     "unitstotal", "lotfront", "lotdepth", "bldgfront", "bldgdepth", "assessland", "assesstot",
                     "exemptland", "exempttot", "builtfar", "xcoord", "ycoord"]
all_columns = []
all_columns.extend(categorical_columns)
all_columns.extend(numerical_columns)

# Fill null entries
for categorical_column in categorical_columns:
    train_data[categorical_column].fillna("Empty", inplace=True)
    test_data[categorical_column].fillna("Empty", inplace=True)
    test_vals = set(train_data[categorical_column].unique().tolist())
    train_vals = set(test_data[categorical_column].unique().tolist())
    if not test_vals.issubset(train_vals):
        not_contained = [t for t in test_vals if not t in train_vals]
        print("{0} {1} not contained.".format(categorical_column, len(not_contained)))
# Select columns; first categoricals then noncategoricals
train_data_ordered = train_data[all_columns].copy()
labels_ordered = train_data["target__office"].copy()
print("X")

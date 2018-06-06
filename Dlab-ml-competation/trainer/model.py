# Copyright 2016 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""Define a Wide + Deep model for classification on structured data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import six
import tensorflow as tf


# Define the format of your input data including unused columns
CSV_COLUMNS = ['Record_ID',
               'Target1',
               'Target2',
               'Target3',
               'Source_System',
               'Product',
               'Underwriting_Year',
               'Effective_Date',
               'Expiry_Date',
               'Transaction_Type',
               'Public_Liability_Limit',
               'Employers_Liability_Limit',
               'Tools_Sum_Insured',
               'Professional_Indemnity_Limit',
               'Contract_Works_Sum_Insured',
               'Hired_in_Plan_Sum_Insured',
               'Own_Plant_Sum_Insured',
               'Trade_1',
               'Trade_2',
               'Manual_EE',
               'Clerical_EE',
               'Subcontractor_EE',
               'Match_Type',
               'Trade_1_Category',
               'Trade_2_Category',
               'Trade_1_Risk Level',
               'Trade_2_Risk Level',
               'Effective_Date2',
               'CancellationEffectiveDate',
               'Commission_Amount',
               'Policy_Count',
               'Gross_PI_Premium',
               'DurationofPolicy',
               'CombinedTradeRiskLevel',
               # 'Public_Liability_Limit_1000000',
               # 'Public_Liability_Limit_1000000_1',
               # 'Public_Liability_Limit_2000000',
               # 'Public_Liability_Limit_5000000',
               # 'Public_Liability_Limit_5000000_1',
               # 'Public_Liability_Limit_1000000_2',
               # 'Public_Liability_Limit_1000000_3',
               # 'Employers_Liability_Limit_1000',
               # 'Professional_Indemnity_Limit_5',
               # 'Professional_Indemnity_Limit_5_1',
               # 'Professional_Indemnity_Limit_1',
               # 'Professional_Indemnity_Limit_1_1',
               # 'Professional_Indemnity_Limit_2',
               # 'Professional_Indemnity_Limit_2_1',
               # 'Professional_Indemnity_Limit_5_2',
               # 'Professional_Indemnity_Limit_5_3',
               # 'Professional_Indemnity_Limit_1_2',
               # 'Professional_Indemnity_Limit_1_3',
               # 'Professional_Indemnity_Limit_2_2',
               # 'Professional_Indemnity_Limit_2_3',
               'Tools_Sum_Insured_Ind',
               'Contract_Works_Sum_Insured_Ind',
               'Hired_in_Plan_Sum_Insured_Ind',
               'Own_Plant_Sum_Insured_Ind',
               'Location',
               # 'Public_Liability_Limit_5000000_2',
               # 'Public_Liability_Limit_5000000_3',
               # 'Professional_Indemnity_Limit_g',
               'Risk_Postcode2',
               'TotalEmployees']


CSV_COLUMN_DEFAULTS = [
    [0],  # 'Record_ID',
    [0],  # 'Target1',
    [0],  # 'Target2',
    [0],  # 'Target3',
    [''],  # 'Source System',
    [''],  # 'Product',
    [0],  # 'Underwriting Year',
    [''],  # 'Effective Date',
    [''],  # 'Expiry Date',
    [''],  # 'Transaction Type',
    [0],  # 'Public Liability Limit',
    [0],  # 'Employers Liability Limit',
    [0],  # 'Tools Sum Insured',
    [0],  # 'Professional Indemnity Limit',
    [0],  # 'Contract Works Sum Insured',
    [0],  # 'Hired in Plan Sum Insured',
    [0],  # 'Own Plant Sum Insured',
    [''],  # 'Trade 1',
    [''],  # 'Trade 2',
    [0],  # 'Manual EE',
    [0],  # 'Clerical EE',
    [0],  # 'Subcontractor EE',
    [0],  # 'Match Type',
    [''],  # 'Trade 1 Category',
    [''],  # 'Trade 2 Category',
    [0],  # 'Trade 1 Risk Level',
    [0],  # 'Trade 2 Risk Level',
    [''],  # 'Effective_Date2',
    [''],  # 'CancellationEffectiveDate',
    [0],  # 'Commission Amount',
    [0],  # 'Policy Count',
    [0],  # 'Gross PI Premium',
    [0],  # 'DurationofPolicy',
    [0],  # 'CombinedTradeRiskLevel',
    # [0],  # 'Public_Liability_Limit_1000000',
    # [0],  # 'Public_Liability_Limit_1000000.1',
    # [0],  # 'Public_Liability_Limit_2000000',
    # [0],  # 'Public_Liability_Limit_5000000',
    # [0],  # 'Public_Liability_Limit_5000000.1',
    # [0],  # 'Public_Liability_Limit_1000000.2',
    # [0],  # 'Public_Liability_Limit_1000000.3',
    # [0],  # 'Employers_Liability_Limit_1000',
    # [0],  # 'Professional_Indemnity_Limit_5',
    # [0],  # 'Professional_Indemnity_Limit_5.1',
    # [0],  # 'Professional_Indemnity_Limit_1',
    # [0],  # 'Professional_Indemnity_Limit_1.1',
    # [0],  # 'Professional_Indemnity_Limit_2',
    # [0],  # 'Professional_Indemnity_Limit_2.1',
    # [0],  # 'Professional_Indemnity_Limit_5.2',
    # [0],  # 'Professional_Indemnity_Limit_5.3',
    # [0],  # 'Professional_Indemnity_Limit_1.2',
    # [0],  # 'Professional_Indemnity_Limit_1.3',
    # [0],  # 'Professional_Indemnity_Limit_2.2',
    # [0], [],  # 'Professional_Indemnity_Limit_2.3',
    [0],  # 'Tools_Sum_Insured_Ind',
    [0],  # 'Contract_Works_Sum_Insured_Ind',
    [0],  # 'Hired_in_Plan_Sum_Insured_Ind',
    [0],  # 'Own_Plant_Sum_Insured_Ind',
    [''],  # 'Location',
    # [0],  # 'Public_Liability_Limit_5000000.2',
    # [0],  # 'Public_Liability_Limit_5000000.3',
    # [0],  # 'Professional_Indemnity_Limit_g',
    [''],  # 'Risk_Postcode2',
    [0]  # 'TotalEmployees'
]
LABEL_COLUMN = 'Target1'
LABELS = [0, 1]

# Define the initial ingestion of each feature used by your model.
# Additionally, provide metadata about the feature.
INPUT_COLUMNS = [
    # Categorical base columns

    # For categorical columns with known values we can provide lists
    # of values ahead of time.
    tf.feature_column.categorical_column_with_vocabulary_list(
        'Source_System', ['Custom', 'Simple']),

    tf.feature_column.categorical_column_with_vocabulary_list(
        'Product',
        ['TradeA', 'TradeB', 'TradeC', 'TradeD', 'TradeE']),


    # categorical_column_with_identity
    tf.feature_column.categorical_column_with_identity(
        'Underwriting_Year',
        num_buckets=5),   # [2010, 2011, 2012, 2013, 2014]
    tf.feature_column.categorical_column_with_identity(
        'Transaction_Type',
        num_buckets=2),   # ['New business', 'Renewal']
    tf.feature_column.categorical_column_with_identity(
        'Public_Liability_Limit',
        num_buckets=4),   # [1000000, 2000000, 5000000, 10000000]
    tf.feature_column.categorical_column_with_identity(
        'Employers_Liability_Limit',
        num_buckets=2),   # [0, 10000000]
    tf.feature_column.categorical_column_with_identity(
        'Professional_Indemnity_Limit',
        num_buckets=7),   # [0, 50000, 100000, 250000, 500000, 1000000, 2000000]
    tf.feature_column.categorical_column_with_identity(
        'Tools_Sum_Insured_Ind',
        num_buckets=2),   # [0, 1]
    tf.feature_column.categorical_column_with_identity(
        'Contract_Works_Sum_Insured_Ind_3',
        num_buckets=2),   # [0, 1]
    tf.feature_column.categorical_column_with_identity(
        'Hired_in_Plan_Sum_Insured_Ind_3',
        num_buckets=2),   # [0, 1]
    tf.feature_column.categorical_column_with_identity(
        'Own_Plant_Sum_Insured_Ind_3',
        num_buckets=2),   # [0, 1]
    tf.feature_column.categorical_column_with_identity(
        'Match_Type',
        num_buckets=8),


    # For columns with a large number of values, or unknown values
    # We can use a hash function to convert to categories.
    tf.feature_column.categorical_column_with_hash_bucket(
        'Trade_1_Category', hash_bucket_size=100, dtype=tf.string),
    tf.feature_column.categorical_column_with_hash_bucket(
        'Location', hash_bucket_size=100, dtype=tf.string),
    tf.feature_column.categorical_column_with_hash_bucket(
        'Risk_Postcode2', hash_bucket_size=100, dtype=tf.string),


    # Continuous base columns.
    tf.feature_column.numeric_column('Tools_Sum_Insured'),
    tf.feature_column.numeric_column('Contract_Works_Sum_Insured'),
    tf.feature_column.numeric_column('Hired_in_Plan_Sum_Insured'),
    tf.feature_column.numeric_column('Own_Plant_Sum_Insured'),
    tf.feature_column.numeric_column('Manual_EE'),
    tf.feature_column.numeric_column('Clerical_EE'),
    tf.feature_column.numeric_column('Subcontractor_EE'),
    tf.feature_column.numeric_column('Trade_1_Risk_Level'),
    tf.feature_column.numeric_column('Trade_2_Risk_Level'),
    tf.feature_column.numeric_column('Commission_Amount'),
    tf.feature_column.numeric_column('Gross_PI_Premium'),
    tf.feature_column.numeric_column('DurationofPolicy'),
    tf.feature_column.numeric_column('CombinedTradeRiskLevel'),
    tf.feature_column.numeric_column('TotalEmployees')
]

UNUSED_COLUMNS = set(CSV_COLUMNS) - {col.name for col in INPUT_COLUMNS} - \
    {LABEL_COLUMN}


def build_estimator(config, embedding_size=8, hidden_units=None):

  """Build a wide and deep model for predicting income category.

  Wide and deep models use deep neural nets to learn high level abstractions
  about complex features or interactions between such features.
  These models then combined the outputs from the DNN with a linear regression
  performed on simpler features. This provides a balance between power and
  speed that is effective on many structured data problems.

  You can read more about wide and deep models here:
  https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html

  To define model we can use the prebuilt DNNCombinedLinearClassifier class,
  and need only define the data transformations particular to our dataset, and
  then
  assign these (potentially) transformed features to either the DNN, or linear
  regression portion of the model.

  Args:
    config: tf.contrib.learn.RunConfig defining the runtime environment for the
      estimator (including model_dir).
    embedding_size: int, the number of dimensions used to represent categorical
      features when providing them as inputs to the DNN.
    hidden_units: [int], the layer sizes of the DNN (input layer first)
    learning_rate: float, the learning rate for the optimizer.
  Returns:
    A DNNCombinedLinearClassifier
  """
  (Source_System,
   Product,
   Underwriting_Year,
   # Effective_Date,
   # Expiry_Date,
   Transaction_Type,
   Public_Liability_Limit,
   Employers_Liability_Limit,
   Tools_Sum_Insured,
   Professional_Indemnity_Limit,
   Contract_Works_Sum_Insured,
   Hired_in_Plan_Sum_Insured,
   Own_Plant_Sum_Insured,
   # Trade_1,
   # Trade_2,
   Manual_EE,
   Clerical_EE,
   Subcontractor_EE,
   Match_Type,
   Trade_1_Category,
   # Trade_2_Category,
   Trade_1_Risk_Level,
   Trade_2_Risk_Level,
   # Effective_Date2,
   # CancellationEffectiveDate,
   Commission_Amount,
   # Policy_Count,
   Gross_PI_Premium,
   DurationofPolicy,
   CombinedTradeRiskLevel,
   # Public_Liability_Limit_1000000,
   # Public_Liability_Limit_1000000_1,
   # Public_Liability_Limit_2000000,
   # Public_Liability_Limit_5000000,
   # Public_Liability_Limit_5000000_1,
   # Public_Liability_Limit_1000000_2,
   # Public_Liability_Limit_1000000_3,
   # Employers_Liability_Limit_1000,
   # Professional_Indemnity_Limit_5,
   # Professional_Indemnity_Limit_5_1,
   # Professional_Indemnity_Limit_1,
   # Professional_Indemnity_Limit_1_1,
   # Professional_Indemnity_Limit_2,
   # Professional_Indemnity_Limit_2_1,
   # Professional_Indemnity_Limit_5_2,
   # Professional_Indemnity_Limit_5_3,
   # Professional_Indemnity_Limit_1_2,
   # Professional_Indemnity_Limit_1_3,
   # Professional_Indemnity_Limit_2_2,
   # Professional_Indemnity_Limit_2_3,
   Tools_Sum_Insured_Ind,
   Contract_Works_Sum_Insured_Ind,
   Hired_in_Plan_Sum_Insured_Ind,
   Own_Plant_Sum_Insured_Ind,
   Location,
   # Public_Liability_Limit_5000000_2,
   # Public_Liability_Limit_5000000_3,
   # Professional_Indemnity_Limit_g,
   Risk_Postcode2,
   TotalEmployees) = INPUT_COLUMNS
  # Build an estimator.

  # Reused Transformations.
  # Continuous columns can be converted to categorical via bucketization
  # age_buckets = tf.feature_column.bucketized_column(
  #     age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  # Wide columns and deep columns.
  wide_columns = [
      # Interactions between different categorical features can also
      # be added as new virtual features.
      # tf.feature_column.crossed_column(
      #     ['education', 'occupation'], hash_bucket_size=int(1e4)),
      # tf.feature_column.crossed_column(
      #     [age_buckets, race, 'occupation'], hash_bucket_size=int(1e6)),
      # tf.feature_column.crossed_column(
      #     ['native_country', 'occupation'], hash_bucket_size=int(1e4)),
      Tools_Sum_Insured,
      Professional_Indemnity_Limit,
      Contract_Works_Sum_Insured,
      Hired_in_Plan_Sum_Insured,
      Own_Plant_Sum_Insured,
      Manual_EE,
      Clerical_EE,
      Subcontractor_EE,
      Trade_1_Category,
      # Trade_2_Category,
      Trade_1_Risk_Level,
      Trade_2_Risk_Level,
      Commission_Amount,
      Gross_PI_Premium,
      DurationofPolicy,
      CombinedTradeRiskLevel,
      TotalEmployees,
  ]


  deep_columns = [
      # Use indicator columns for low dimensional vocabularies
      tf.feature_column.indicator_column(Source_System),
      tf.feature_column.indicator_column(Product),
      tf.feature_column.indicator_column(Transaction_Type),
      tf.feature_column.indicator_column(Underwriting_Year),
      tf.feature_column.indicator_column(Public_Liability_Limit),
      tf.feature_column.indicator_column(Employers_Liability_Limit),

      tf.feature_column.indicator_column(Tools_Sum_Insured_Ind),
      tf.feature_column.indicator_column(Contract_Works_Sum_Insured_Ind),
      tf.feature_column.indicator_column(Hired_in_Plan_Sum_Insured_Ind),
      tf.feature_column.indicator_column(Own_Plant_Sum_Insured_Ind),

      # Use embedding columns for high dimensional vocabularies
      tf.feature_column.embedding_column(
          Location, dimension=embedding_size),
      tf.feature_column.embedding_column(
          Risk_Postcode2, dimension=embedding_size),
      tf.feature_column.embedding_column(
          Trade_1_Category, dimension=embedding_size),
  ]


  return tf.estimator.DNNLinearCombinedClassifier(
      config=config,
      linear_feature_columns=wide_columns,
      dnn_feature_columns=deep_columns,
      dnn_hidden_units=hidden_units or [100, 70, 50, 25]
  )


def parse_label_column(label_string_tensor):
  """Parses a string tensor into the label tensor
  Args:
    label_string_tensor: Tensor of dtype string. Result of parsing the
    CSV column specified by LABEL_COLUMN
  Returns:
    A Tensor of the same shape as label_string_tensor, should return
    an int64 Tensor representing the label index for classification tasks,
    and a float32 Tensor representing the value for a regression task.
  """
  # Build a Hash Table inside the graph
  table = tf.contrib.lookup.index_table_from_tensor(tf.constant(LABELS))

  # Use the hash table to convert string labels to ints and one-hot encode
  return table.lookup(label_string_tensor)


# ************************************************************************
# YOU NEED NOT MODIFY ANYTHING BELOW HERE TO ADAPT THIS MODEL TO YOUR DATA
# ************************************************************************


def csv_serving_input_fn():
  """Build the serving inputs."""
  csv_row = tf.placeholder(
      shape=[None],
      dtype=tf.string
  )
  features = parse_csv(csv_row)
  features.pop(LABEL_COLUMN)
  return tf.estimator.export.ServingInputReceiver(features, {'csv_row': csv_row})


def example_serving_input_fn():
  """Build the serving inputs."""
  example_bytestring = tf.placeholder(
      shape=[None],
      dtype=tf.string,
  )
  feature_scalars = tf.parse_example(
      example_bytestring,
      tf.feature_column.make_parse_example_spec(INPUT_COLUMNS)
  )
  return tf.estimator.export.ServingInputReceiver(
      features,
      {'example_proto': example_bytestring}
  )

# [START serving-function]
def json_serving_input_fn():
  """Build the serving inputs."""
  inputs = {}
  for feat in INPUT_COLUMNS:
    inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)
    
  return tf.estimator.export.ServingInputReceiver(inputs, inputs)
# [END serving-function]

SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'EXAMPLE': example_serving_input_fn,
    'CSV': csv_serving_input_fn
}


def parse_csv(rows_string_tensor):
  """Takes the string input tensor and returns a dict of rank-2 tensors."""

  # Takes a rank-1 tensor and converts it into rank-2 tensor
  # Example if the data is ['csv,line,1', 'csv,line,2', ..] to
  # [['csv,line,1'], ['csv,line,2']] which after parsing will result in a
  # tuple of tensors: [['csv'], ['csv']], [['line'], ['line']], [[1], [2]]
  row_columns = tf.expand_dims(rows_string_tensor, -1)
  columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
  features = dict(zip(CSV_COLUMNS, columns))

  # Remove unused columns
  for col in UNUSED_COLUMNS:
    features.pop(col)
  return features


def input_fn(filenames,
                      num_epochs=None,
                      shuffle=True,
                      skip_header_lines=0,
                      batch_size=200):
  """Generates features and labels for training or evaluation.
  This uses the input pipeline based approach using file name queue
  to read data so that entire data is not loaded in memory.

  Args:
      filenames: [str] list of CSV files to read data from.
      num_epochs: int how many times through to read the data.
        If None will loop through data indefinitely
      shuffle: bool, whether or not to randomize the order of data.
        Controls randomization of both file order and line order within
        files.
      skip_header_lines: int set to non-zero in order to skip header lines
        in CSV files.
      batch_size: int First dimension size of the Tensors returned by
        input_fn
  Returns:
      A (features, indices) tuple where features is a dictionary of
        Tensors, and indices is a single Tensor of label indices.
  """
  filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
  if shuffle:
    # Process the files in a random order.
    filename_dataset = filename_dataset.shuffle(len(filenames))
    
  # For each filename, parse it into one element per line, and skip the header
  # if necessary.
  dataset = filename_dataset.flat_map(
      lambda filename: tf.data.TextLineDataset(filename).skip(skip_header_lines))
  
  dataset = dataset.map(parse_csv)
  if shuffle:
    dataset = dataset.shuffle(buffer_size=batch_size * 10)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  features = iterator.get_next()
  return features, parse_label_column(features.pop(LABEL_COLUMN))

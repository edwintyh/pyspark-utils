import pyspark
from pyspark.sql.window import Window
from pyspark.sql import functions as f

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('spark-session').getOrCreate()

def shape(self):
  
  '''
  Find the number of rows and columns in a DataFrame
  
  '''
  
  return self.count(), len(self.schema.names)

pyspark.sql.DataFrame.shape = shape



def info(self, show = True):
  
  '''
  Print concise summary of a pyspark.sql.DataFrame
  This method prints information about a DataFrame
  including the index dtype and columns, non-null values
  
  Args:
    show(bool): default True. show result
    
    
  Returns:
    pyspark.sql.DataFrame
  
  '''
  
  subset = self.schema.names
  total_rows = self.count()
  _non_null = \
    self.select([(total_rows - f.sum(f.when(f.col(col).isNull(),1).otherwise(0))).alias(col) for col in subset])\
    .toPandas()\
    .transpose()\
    .reset_index()\
    .rename(columns={'index':'Column', 0:'Non-Null Count'})
  _non_null = spark.createDataFrame(_non_null)
  _dtype = spark.createDataFrame(self.dtypes).withColumnRenamed('_1','Column').withColumnRenamed('_2','Dtype')
  result = _dtype.join(_non_null, on = 'Column').select('Column', 'Non-Null Count', 'Dtype')
  
  if show:
    return result.show()
  else:
    return result

pyspark.sql.DataFrame.info = info


def rename(self, columns):
  
  '''
  Rename colum headers
  
  Args:
    columns(dict): 
      A dictionary where key are current column names and values are new column names
      {'old_column_name1':'new_column_name1', 'old_column_name2':'new_column_name2'}
    
  Returns:
    pyspark.sql.DataFrame
  
  
  '''
  
  for old_name, new_name in columns.items():
    self = self.withColumnRenamed(old_name, new_name)

  return self


pyspark.sql.DataFrame.rename = rename

def valueCounts(self, subset, normalize = False, sort = True, ascending = False, show = True):
  
  
  '''
  Count of unique rows in a DataFrame
  
  Args:
  
    subset(list): column to be used when counting.
    normalize(bool): default False. Return proportion instead of frequencies.
    sort(bool): default False. Sort by frequencies.
    ascending(bool): default False. Sort in ascending order.
    
    
  Return:
    pyspark.sql.DataFrame
  
  '''
  
  w = Window.partitionBy().rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
  
  self = \
  self\
    .groupby(subset)\
    .count()
  
  if sort:
    self = self.sort('count', ascending = ascending)
  else:
    self = self.sort(subset)
    
  
  if normalize:
    self = \
    self\
    .withColumn('pct', f.round(f.col('count')/f.sum('count').over(w),4))\
    .drop('count')
  
  if show:
    return self.show()
  else:
    return self

pyspark.sql.DataFrame.value_counts = valueCounts

def duplicated(self, subset = None, orderby = None, ascending = False, keep = 'first'):
  
  '''
  Returns pyspark.sql.DataFrame with duplicate indicator column. True = duplicate(s)
  
  Args:
    subset(list):
      default None. list of column for identifying duplicates.
      Default uses all the columns.
      
    orderby(list): default None. list of column to order by.
      Required if keep is first or last. Uses subset if None.
      
    ascending(bool,list of bool): default False. Order in ascending or descending
      sequence. Required if keep is first or last.
      
    keep ({'first', 'last', False}):
      default 'first'.
      Determines which duplicates (if any) to mark.
      first : Mark duplicates as True except for the first occurrence.
      last : Mark duplicates as True except for the last occurrence.
      False : Mark all duplicates as True.
      
  Returns:
    self(pyspark.sql.DataFrame)
  
  '''
  
  if subset == None:
    subset = self.schema.names
    
  subset = [subset] if isinstance(subset, str) else subset

  assert keep in ['first', 'last', False], 'keep must be either first, last or False'
  
  if orderby == None:
    orderby = subset
  elif isinstance(orderby, str):
    orderby = [orderby]   
    
  if isinstance(ascending, bool):
    if ascending:
      ordering = [f.asc] * len(orderby)
    else:
      ordering = [f.desc] * len(orderby)
  
  elif isinstance(ascending, list):
    assert all([isinstance(i, bool) for i in ascending]), 'ascending should be bool or list of bool'
    ordering = [f.asc if i else f.desc for i in ascending]
    
  w1 = Window.partitionBy(*subset).orderBy(*[ordering[idx](i) for idx, i in enumerate(orderby)]).rowsBetween(Window.unboundedPreceding, Window.currentRow)
  w2 = Window.partitionBy(*subset).orderBy(*[ordering[idx](i) for idx, i in enumerate(orderby)]).rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
     
  self = \
    self\
    .sort(orderby, ascending = ascending)\
    .withColumn('seq', f.row_number().over(w1))
  
  if keep == 'first':
    self = self.withColumn('duplicate_indicator', f.when(f.col('seq') == 1, False).otherwise(True)).drop(*['seq'])
    
  elif keep == 'last':
    self = \
      self\
      .withColumn('max_seq', f.max('seq').over(w2))\
      .withColumn('duplicate_indicator', f.when(f.col('seq') == f.col('max_seq'), False).otherwise(True))\
      .drop(*['seq', 'max_seq'])
    
  else:
    self = \
      self\
      .withColumn('max_seq', f.max('seq').over(w2))\
      .withColumn('duplicate_indicator', f.when(f.col('max_seq') > 1, True).otherwise(False)).drop(*['seq', 'max_seq'])
    
  return self


pyspark.sql.DataFrame.duplicated = duplicated

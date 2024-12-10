import os 
import argparse
import yaml
import luigi 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder




class File(luigi.ExternalTask):
	file = luigi.Parameter()

	def output(self):
		return luigi.LocalTarget(self.file)


class OneToOneTask(luigi.Task):
	input_file = luigi.Parameter() 
	output_file = luigi.Parameter() 
	params = luigi.DictParameter() 
	print(params)
	def convert_csv_to_df(self):
		df = pd.read_csv(self.input().path, index_col="Datetime", parse_dates=True)
		df.replace('#VALUE!', np.nan, inplace=True)
		df.replace([np.inf, -np.inf], np.nan, inplace=True)
		df = df.apply(pd.to_numeric, errors='coerce')
		return df
	def convert_df_to_csv(self,dataframe):
		dataframe.to_csv(self.output().path, index = True)
	def find_important_feature(self,df, target_column):
		Correlation_matrix = df.corr().loc[target_column].sort_values(ascending = False)
		important_feature = Correlation_matrix.index[1]
		return important_feature
	def run(self):
		pass 

	def get_input_format(self):
		pass 

	def get_output_format(self):
		pass 
	
class checkDataset(OneToOneTask):
	def requires(self):
		return File(file=self.input_file)
	def output(self):
		return luigi.LocalTarget(self.output_file)

	
	def run(self):
		df = pd.read_csv(self.input().path)
		if 'Datetime' not in df.columns:
			raise ValueError(f"The dataset at {self.input().path} does not contain a 'Datetime' column.")
		df = self.convert_csv_to_df()
		self.convert_df_to_csv(df)
		df.info()
		

class calculateLags(OneToOneTask):
	def requires(self):
		return checkDataset(input_file=self.input_file, output_file=self.output_file, params=self.params)
	def output(self):
		return luigi.LocalTarget(self.output_file)

	
	def run(self):
		print("Inside lags dataset")
		target_column=self.params['target_column']
		lag_sizes = self.params['lags_size']
		print("Inside lags dataset")
		df = self.convert_csv_to_df()
		important_feature = self.find_important_feature(df,target_column)
		print(important_feature)
		for lag_size in lag_sizes:
			lagged_column_name = f'lag_{lag_size}_{important_feature}'
			df[lagged_column_name] = df[important_feature].shift(lag_size)
			df[f'lag_{lag_size}_{important_feature}'] = df[important_feature].shift(lag_size)        
		print(f"Saving processed lagged dataset to: {self.output().path}")
		self.convert_df_to_csv(df)

class calculateRollingWindow(OneToOneTask):
	def requires(self):
		return checkDataset(input_file=self.input_file, output_file=self.output_file, params=self.params)
	def output(self):
		return luigi.LocalTarget(self.output_file)	
	def run(self):
		print("Inside roll window ")
		print(self.input().path)
		target_column=self.params['target_column']
		window_size = self.params['window_size']
		start = self.params['start']
		df = self.convert_csv_to_df()
		important_feature = self.find_important_feature(df, target_column)
		df[f'rolling_mean_{start}_{window_size}'] = df[important_feature].rolling(window=window_size).mean()
		self.convert_df_to_csv(df)

class calculateHoursofDay(OneToOneTask):
	def requires(self):
		return checkDataset(input_file=self.input_file, output_file=self.output_file, params=self.params)
	def output(self):
		return luigi.LocalTarget(self.output_file)	
	def run(self):
		column = self.params['hour_column']
		print("Inside hours of day ")
		df = self.convert_csv_to_df()
		df[column] = df.index.hour
		self.convert_df_to_csv(df)

class calculateSineCosaineHour(OneToOneTask):
	def requires(self):
		return calculateHoursofDay(input_file=self.input_file, output_file=self.output_file, params=self.params)
		
	def output(self):
		return luigi.LocalTarget(self.output_file)	
	def run(self):
		hour_column = self.params['hour_column']
		sine_column = self.params['sine_column']
		cosaine_column = self.params['cos_column']
		print("Inside sine and cos ")
		df = self.convert_csv_to_df()
		df[sine_column] = np.sin(df[hour_column] / 24 * 2 * np.pi)
		df[cosaine_column] = np.cos(df[hour_column] / 24.0 * 2 * np.pi)
		self.convert_df_to_csv(df)
class calculateMinMaxScaler(OneToOneTask):
	def requires(self):
		return checkDataset(input_file=self.input_file, output_file=self.output_file, params=self.params)

	def output(self):
		return luigi.LocalTarget(self.output_file)	
	def run(self):
		print("Inside minmax scaler ")
		df = self.convert_csv_to_df()
		scaler = MinMaxScaler()
		numeric_columns = df.select_dtypes(include=[np.number]).columns
		df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
		self.convert_df_to_csv(df)
class attributeAddr(OneToOneTask):
	def requires(self):
		return checkDataset(input_file=self.input_file, output_file=self.output_file, params=self.params)
	def output(self):
		return luigi.LocalTarget(self.output_file)	
	def run(self):
		target_column = self.params['target_column']
		test_number = self.params['test_number']
		df = self.convert_csv_to_df()
		correlation_matrix = df.corr().loc[target_column]
		list2=[]
		for i in range(len(correlation_matrix)):
			if correlation_matrix[i] <0.15 and correlation_matrix[i]>-0.15:
				list2.append(correlation_matrix.index[i])
		print("list 2")
		print(list2)
		if list2:
			for i in range(test_number):
				index1 = np.random.randint(len(list2))
				print(index1)
				index2 = np.random.randint(len(list2))
				x = list2[index1]
				y = list2[index2]
				z = x+ "_per_"+ y
				test_attributes =  pd.DataFrame()
				test_attributes[target_column] = df[target_column]
				test_attributes[z] = df[x] /df[y]
				correlation_with_target = test_attributes.corr().loc[z][target_column]
				if correlation_with_target >0.15 or correlation_with_target <-0.15:
					print(z + " have a pretty high correlation "+ str(correlation_with_target) +" with target")
					df[z] = test_attributes[z]
				else:
					print(z + "is not much suitable")

		self.convert_df_to_csv(df)
class transformStandardScaler(OneToOneTask):
	def requires(self):
		return checkDataset(input_file=self.input_file, output_file=self.output_file, params=self.params)

	def output(self):
		return luigi.LocalTarget(self.output_file)	
	def run(self):
		target_column = self.params['target_column']
		print("Inside standard scaler ")
		df = self.convert_csv_to_df()
		features = df.drop(columns=[target_column])
		target = df[target_column]
		columns = features.columns
		index = features.index
		scaler =  StandardScaler()		
		scaled_data = scaler.fit_transform(features)
		scaled_df = pd.DataFrame(scaled_data, columns=columns, index=index)
		final_df = pd.concat([scaled_df, target], axis=1)
		self.convert_df_to_csv(final_df)

class doPca(OneToOneTask):
	def requires(self):
		return checkDataset(input_file=self.input_file, output_file=self.output_file, params=self.params)

	def output(self):
		return luigi.LocalTarget(self.output_file)	
	def run(self):
		target_column = self.params['target_column']
		n_components = self.params['n_components']
		reduction = self.params['reduction']

		print("Inside pca ")
		df = self.convert_csv_to_df()
		#target_column_data = df[target_column]
		features = df.drop(columns=[target_column]) if target_column in df.columns else df
		pca = PCA(n_components=n_components)
		pca_data = pca.fit_transform(features)
		feature_names = features.columns
		pca_column_names = [f'PC{i+1}_{feature}' for i, feature in enumerate(feature_names[:n_components])]
		pca_df = pd.DataFrame(pca_data, columns=pca_column_names, index=df.index)
		if reduction == 1:
			full_df = pca_df
		else: 
			full_df = df.join(pca_df)
		if target_column in df.columns:
			full_df[target_column] = df[target_column]
		self.convert_df_to_csv(full_df)

	
class fillMissingValuesImputer(OneToOneTask):
	def requires(self):
		return checkDataset(input_file=self.input_file, output_file=self.output_file, params=self.params)

	def output(self):
		return luigi.LocalTarget(self.output_file)	
	def run(self):
		print("Inside imputer")
		df = self.convert_csv_to_df()
		numeric_df = df.select_dtypes(include=[np.number])
		columns = numeric_df.columns
		index = numeric_df.index
	
		strategy = self.params['strategy']
		imputer = SimpleImputer(strategy = strategy)
		imputed_data = imputer.fit_transform(numeric_df)
		imputed_pd = pd.DataFrame(imputed_data,columns=columns, index=index)
		non_numeric_df = df.select_dtypes(exclude=[np.number])
		result_df = pd.concat([imputed_pd, non_numeric_df], axis=1)

		self.convert_df_to_csv(result_df)

class fillMissingValuesbfill(OneToOneTask):
	def requires(self):
		return checkDataset(input_file=self.input_file, output_file=self.output_file, params=self.params)

	def output(self):
		return luigi.LocalTarget(self.output_file)	
	def run(self):
		print("Inside bfill")
		df = self.convert_csv_to_df()
		df.fillna(method='bfill', inplace= True)
		self.convert_df_to_csv(df)



class doOneHotEncoding(OneToOneTask):
	def requires(self):
		return checkDataset(input_file=self.input_file, output_file=self.output_file, params=self.params)

	def output(self):
		return luigi.LocalTarget(self.output_file)	
	def run(self):
		df = self.convert_csv_to_df()
		df['year'] = df.index.year
		df['month'] = df.index.month
		df['day'] = df.index.day
		df['hour'] = df.index.hour
		datetime_columns = ['year', 'month', 'day', 'hour'] # e.g., ['year', 'month', 'day', 'hour']
		encoder = OneHotEncoder(sparse_output=False)
		encoded_data = encoder.fit_transform(df[datetime_columns])
		encoded_column_names = encoder.get_feature_names_out(datetime_columns)
		encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoded_column_names,
            index=df.index
        )
		df = pd.concat([df.drop(columns=datetime_columns), encoded_df], axis=1)
		self.convert_df_to_csv(df)
class PreprocessingPipeline(luigi.WrapperTask):	
	input_dir = luigi.Parameter() 
	output_dir = luigi.Parameter()
	config = luigi.Parameter() 
	tasks = {
		'check_dataset': checkDataset,
		'calculate_lags': calculateLags,
		'calculate_rolling_window': calculateRollingWindow,
		'calculate_hours_of_day': calculateHoursofDay,
		'calculate_sine_cosaine': calculateSineCosaineHour,
		'minmaxscaler': calculateMinMaxScaler,
		'attribute_adder': attributeAddr,
		'standard_scaler': transformStandardScaler,
		'imputer': fillMissingValuesImputer,
		'bfill': fillMissingValuesbfill,
		'pca' : doPca,
		'one_hot_encoding' : doOneHotEncoding


	}
	task_mapping = {
		'calculate_lags': ['csv', 'csv'],
		'calculate_rolling_window': ['csv', 'csv'],
		'calculate_hours_of_day': ['csv', 'csv'],
		'calculate_sine_cosaine': ['csv', 'csv'],
		'minmaxscaler': ['csv', 'csv'],
		'attribute_adder':  ['csv', 'csv'],
		'standard_scaler': ['csv', 'csv'],
		'imputer': ['csv', 'csv'],
		'bfill': ['csv', 'csv'],
		'pca': ['csv', 'csv'],
		'one_hot_encoding' : ['csv', 'csv']
	
	}
	
	def requires(self):
		config = self.parse_config()
		pipeline_config = config['pipeline']

		for task in pipeline_config:
			# Task Specifications
			task_type = task['task']
			unique_id = task['id']
			input_id = task['input_id']
			params = task['parameters']
			
			# Input and Output Directories
			input_dir = self.input_dir if input_id == 'input' else os.path.join(self.output_dir, input_id)
			output_dir = os.path.join(self.output_dir, unique_id)
			
			# Generating output tree 
			input_tree = self.get_directory_tree(input_dir)
			for subdir in input_tree:
				os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

			# Input and output files
			input_format, output_format = self.get_io_format(task)
			input_files = self.get_files(input_dir, input_format)
			output_files = []
			for file in input_files:
				file = os.path.relpath(file, input_dir)
				file, ext = os.path.splitext(file)
				file = file + "." + output_format 
				output_files.append(os.path.join(output_dir, file))
			mapping = list(zip(input_files, output_files))

			# Instantiating Tasks
			pending_tasks = []
			for input_file, output_file in mapping:
				pending_tasks.append(
					self.tasks[task_type](
						input_file=input_file, 
						output_file=output_file,
						params=params
					)
				)

			yield pending_tasks

	def get_directory_tree(self, directory):
		level = 0 
		tree = []
		for root, directories, files in os.walk(directory):
			if level > 0:
				relative_path = os.path.relpath(root, directory)
				tree.append(relative_path)
			level += 1
		return tree

	def get_files(self, directory, file_format):
		target_files = [] 
		print(directory)
		for root, directories, files in os.walk(directory):
			print(f"Found files: {files} in directory: {root}")
			for file in files:
				print(file)
				filename, ext = os.path.splitext(file)
				print(filename)
				ext = ext.strip(".")
				if ext == file_format:
					file = os.path.join(root, file)
					target_files.append(file)
		return target_files

	def get_io_format(self, task):
		if task['task'] == 'check_dataset':
			input_format = task['parameters']['input_format']
			output_format = task['parameters']['output_format']
		else:
			input_format, output_format = self.task_mapping[task['task']]
		return input_format, output_format
	def parse_config(self):
		with open(self.config, 'r') as f:
		    return yaml.safe_load(f)
	



if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('-I', '--input', type=str, help='Input Directory Path')
	arg_parser.add_argument('-O', '--output', type=str, help='Output Directory Path')
	arg_parser.add_argument('-C', '--config', type=str, help='Configuration File Path')
	args = arg_parser.parse_args()
	print(args.input)

	luigi.build(
		[PreprocessingPipeline(
			input_dir=args.input, 
			output_dir=args.output, 
			config=args.config
		)], 
		scheduler_host='localhost', 
		scheduler_port=8082
	)

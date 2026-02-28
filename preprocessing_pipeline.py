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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

class File(luigi.ExternalTask):
	file = luigi.Parameter()

	def output(self):
		return luigi.LocalTarget(self.file)


class OneToOneTask(luigi.Task):
	input_file = luigi.Parameter() 
	output_file = luigi.Parameter() 
	params = luigi.DictParameter(default={}) 
	print(params)
	def convert_csv_to_df(self):
		df = pd.read_csv(self.input().path, index_col="Datetime", parse_dates=True)
		df.replace('#VALUE!', np.nan, inplace=True)
		df.replace([np.inf, -np.inf], np.nan, inplace=True)
		df = df.apply(pd.to_numeric, errors='coerce')
		return df
	def convert_df_to_csv(self,dataframe):
		dataframe.to_csv(self.output().path, index = True)
	def find_important_feature(self,df, target_column, mode):
		if mode == "binary":
			Correlation_matrix = df.corr().loc[target_column].sort_values(ascending = False)
			important_feature = Correlation_matrix.index[1]
			return important_feature
		y= df[target_column].astype(str)
		X_df = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
		y_enc = LabelEncoder().fit_transform(y)
		mi = mutual_info_classif(X_df.values, y_enc, random_state=42)
		return X_df.columns[int(np.argmax(mi))]
	def run(self):
		pass 

	def get_input_format(self):
		pass 

	def get_output_format(self):
		pass 

class stratifiedSplit(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		output_files = {}
		file_base = os.path.splitext(os.path.basename(self.input_file[0]))[0]
		for fold in range(1, 6): 
			output_files[f'X_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_X_train_fold_{fold}.csv"))
			output_files[f'Y_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_Y_train_fold_{fold}.csv"))
			output_files[f'X_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_X_test_fold_{fold}.csv"))
			output_files[f'Y_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_Y_test_fold_{fold}.csv"))
		return output_files
	def run(self):
		target_column = self.params['target_column']
		mode = self.params['mode']
		df = pd.read_csv(self.input()[0].path, index_col="Datetime", parse_dates=True)
		if mode == "binary":
			correlation_matrix = df.corr().loc[target_column].sort_values(ascending = False)
			important_feature = correlation_matrix.index[1]
			print("important feature", important_feature)
			devise_metric = df[important_feature].mean() / df[important_feature].std()
			category_count1 = np.int64(df[important_feature].mean() + df[important_feature].std())
			category_count2 = np.int64(df[important_feature].mean() - df[important_feature].std())
			df[important_feature + "_cat"] = np.ceil(df[important_feature] / devise_metric)
			df[important_feature + "_cat"].where(df[important_feature + "_cat"] < category_count1, category_count1, inplace=True)
			df[important_feature + "_cat"].where(df[important_feature + "_cat"] > category_count2, category_count2, inplace=True)
			skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
			for fold, (train_index, test_index) in enumerate(skf.split(df, df[important_feature + "_cat"]), start=1):
				train_dates = df.index[train_index]
				test_dates = df.index[test_index]
				strat_train_set = df.loc[train_dates].drop(columns=important_feature + "_cat")
				strat_test_set = df.loc[test_dates].drop(columns=important_feature + "_cat")
				X_train = strat_train_set.drop(columns=[target_column])
				Y_train = strat_train_set[target_column]
				X_test = strat_test_set.drop(columns=[target_column])
				Y_test = strat_test_set[target_column]
				output_files = self.output()
				X_train.to_csv(output_files[f'X_train_fold_{fold}'].path, index=True)
				Y_train.to_csv(output_files[f'Y_train_fold_{fold}'].path, index=True)
				X_test.to_csv(output_files[f'X_test_fold_{fold}'].path, index=True)
				Y_test.to_csv(output_files[f'Y_test_fold_{fold}'].path, index=True)
		
		else:
			skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
			important_feature = target_column
			for fold, (train_index, test_index) in enumerate(skf.split(df, df[important_feature]), start=1):
				train_dates = df.index[train_index]
				test_dates = df.index[test_index]
				strat_train_set = df.loc[train_dates]
				strat_test_set = df.loc[test_dates]
				X_train = strat_train_set.drop(columns=[target_column])
				Y_train = strat_train_set[target_column]
				X_test = strat_test_set.drop(columns=[target_column])
				Y_test = strat_test_set[target_column]
				output_files = self.output()
				X_train.to_csv(output_files[f'X_train_fold_{fold}'].path, index=True)
				Y_train.to_csv(output_files[f'Y_train_fold_{fold}'].path, index=True)
				X_test.to_csv(output_files[f'X_test_fold_{fold}'].path, index=True)
				Y_test.to_csv(output_files[f'Y_test_fold_{fold}'].path, index=True)


class timeSeriesSplit(OneToOneTask):
    def requires(self):
        return [File(file=f) for f in self.input_file]
    
    def output(self):
        output_files = {}
        file_base = os.path.splitext(os.path.basename(self.input_file[0]))[0]
        for fold in range(1, 6):  
            output_files[f'X_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_X_train_fold_{fold}.csv"))
            output_files[f'Y_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_Y_train_fold_{fold}.csv"))
            output_files[f'X_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_X_test_fold_{fold}.csv"))
            output_files[f'Y_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_Y_test_fold_{fold}.csv"))
        return output_files

    def run(self):
        target_column = self.params['target_column']
        df = pd.read_csv(self.input()[0].path, index_col="Datetime", parse_dates=True)

        
        df = df.sort_index()

        
        tscv = TimeSeriesSplit(n_splits=5)

        for fold, (train_index, test_index) in enumerate(tscv.split(df), start=1):
            train_dates = df.index[train_index]
            test_dates = df.index[test_index]
            
            strat_train_set = df.loc[train_dates]
            strat_test_set = df.loc[test_dates]
            
            X_train = strat_train_set.drop(columns=[target_column])
            Y_train = strat_train_set[target_column]
            X_test = strat_test_set.drop(columns=[target_column])
            Y_test = strat_test_set[target_column]
            
            output_files = self.output()
            
            X_train.to_csv(output_files[f'X_train_fold_{fold}'].path, index=True)
            Y_train.to_csv(output_files[f'Y_train_fold_{fold}'].path, index=True)
            X_test.to_csv(output_files[f'X_test_fold_{fold}'].path, index=True)
            Y_test.to_csv(output_files[f'Y_test_fold_{fold}'].path, index=True)
"""
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
"""		

class calculateHistory(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		output_files = {}
		for input_path in self.input_file:			
			filename = os.path.basename(input_path)
			if '_X_train_fold_' in filename:
				fold = filename.split('_X_train_fold_')[-1].split('.')[0]
				output_files[f'X_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_X_test_fold_' in filename:
				fold = filename.split('_X_test_fold_')[-1].split('.')[0]
				output_files[f'X_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_Y_train_fold_' in filename:
				fold = filename.split('_Y_train_fold_')[-1].split('.')[0]
				output_files[f'Y_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_Y_test_fold_' in filename:
				fold = filename.split('_Y_test_fold_')[-1].split('.')[0]
				output_files[f'Y_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
		return output_files
	
	def run(self):
		print("Inside lags dataset")
		target_column=self.params['target_column']
		mode = self.params['mode']
		lag_sizes = self.params['lags_size']
		window_start = self.params['window_start']
		window_size = self.params['window_size']
		print("Inside history")
		output_files = self.output()

		for fold in range(1, 6):
			
			train_key = f'X_train_fold_{fold}'
			test_key = f'X_test_fold_{fold}'

			if train_key not in output_files or test_key not in output_files:
				continue

			
			df_train = pd.read_csv([f.path for f in self.input() if f'_X_train_fold_{fold}.csv' in f.path][0], index_col="Datetime", parse_dates=True)
			df_test = pd.read_csv([f.path for f in self.input() if f'_X_test_fold_{fold}.csv' in f.path][0], index_col="Datetime", parse_dates=True)
			df_y_train = pd.read_csv([f.path for f in self.input() if f'_Y_train_fold_{fold}.csv' in f.path][0], index_col="Datetime", parse_dates=True)
			df_train_full = df_train.copy()
			df_train_full[target_column] = df_y_train[target_column]

			
			important_feature = self.find_important_feature(df_train_full, target_column,mode)
			print(f"Important feature for fold {fold}: {important_feature}")

			
			for lag_size in lag_sizes:
				df_train[f'lag_{lag_size}_{important_feature}'] = df_train[important_feature].shift(lag_size)
				df_test[f'lag_{lag_size}_{important_feature}'] = df_test[important_feature].shift(lag_size)
				df_train[f'lag_{lag_size}_{important_feature}'].fillna(0, inplace=True)
				df_test[f'lag_{lag_size}_{important_feature}'].fillna(0, inplace=True)


			df_train[f'rolling_mean_{window_start}_{window_size}'] = df_train[important_feature].rolling(window=window_size).mean()
			df_test[f'rolling_mean_{window_start}_{window_size}'] = df_test[important_feature].rolling(window=window_size).mean()
			df_train[f'rolling_mean_{window_start}_{window_size}'].fillna(0, inplace=True)
			df_test[f'rolling_mean_{window_start}_{window_size}'].fillna(0, inplace=True)

			
			df_train.to_csv(output_files[train_key].path, index=True)
			df_test.to_csv(output_files[test_key].path, index=True)

			
			for y_key in [f'Y_train_fold_{fold}', f'Y_test_fold_{fold}']:
				y_input_path = [f.path for f in self.input() if y_key in f.path][0]
				df_y = pd.read_csv(y_input_path, index_col="Datetime", parse_dates=True)
				df_y.to_csv(output_files[y_key].path, index=True)
"""
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

"""
class calculateSineCosaineHour(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		output_files = {}
		for input_path in self.input_file:
			filename = os.path.basename(input_path)
			if '_X_train_fold_' in filename:
				fold = filename.split('_X_train_fold_')[-1].split('.')[0]
				output_files[f'X_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_X_test_fold_' in filename:
				fold = filename.split('_X_test_fold_')[-1].split('.')[0]
				output_files[f'X_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_Y_train_fold_' in filename:
				fold = filename.split('_Y_train_fold_')[-1].split('.')[0]
				output_files[f'Y_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_Y_test_fold_' in filename:
				fold = filename.split('_Y_test_fold_')[-1].split('.')[0]
				output_files[f'Y_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
		return output_files	
	def run(self):
		hour_column = self.params['hour_column']
		sine_column = self.params['sine_column']
		cosaine_column = self.params['cos_column']
		output_files = self.output()
		for fold in range(1, 6):
			input_train_X = [f.path for f in self.input() if f'_X_train_fold_{fold}.csv' in f.path][0]
			input_test_X = [f.path for f in self.input() if f'_X_test_fold_{fold}.csv' in f.path][0]
			input_train_Y = [f.path for f in self.input() if f'_Y_train_fold_{fold}.csv' in f.path][0]
			input_test_Y = [f.path for f in self.input() if f'_Y_test_fold_{fold}.csv' in f.path][0]
			df_train_X = pd.read_csv(input_train_X, index_col="Datetime", parse_dates=True)
			df_test_X = pd.read_csv(input_test_X, index_col="Datetime", parse_dates=True)
			df_train_Y = pd.read_csv(input_train_Y, index_col="Datetime", parse_dates=True)
			df_test_Y = pd.read_csv(input_test_Y, index_col="Datetime", parse_dates=True)
			for x_type, df_X in [('X_train', df_train_X), ('X_test', df_test_X)]:
				df_X[hour_column] = df_X.index.hour
				df_X[sine_column] = np.sin(df_X[hour_column] / 24 * 2 * np.pi)
				df_X[cosaine_column] = np.cos(df_X[hour_column] / 24 * 2 * np.pi)
				df_X.to_csv(output_files[f'{x_type}_fold_{fold}'].path, index=True)
			df_train_Y.to_csv(output_files[f'Y_train_fold_{fold}'].path, index=True)
			df_test_Y.to_csv(output_files[f'Y_test_fold_{fold}'].path, index=True)
	

class attributeAddr(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		output_files = {}
		for input_path in self.input_file:
			filename = os.path.basename(input_path)
			if '_X_train_fold_' in filename:
				fold = filename.split('_X_train_fold_')[-1].split('.')[0]
				output_files[f'X_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_X_test_fold_' in filename:
				fold = filename.split('_X_test_fold_')[-1].split('.')[0]
				output_files[f'X_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_Y_train_fold_' in filename:
				fold = filename.split('_Y_train_fold_')[-1].split('.')[0]
				output_files[f'Y_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_Y_test_fold_' in filename:
				fold = filename.split('_Y_test_fold_')[-1].split('.')[0]
				output_files[f'Y_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
		return output_files	
	def run(self):
		target_column = self.params['target_column']
		test_number = self.params['test_number']
		output_files = self.output()
		
		for fold in range(1, 6):
			input_train_X = [f.path for f in self.input() if f'_X_train_fold_{fold}.csv' in f.path][0]
			input_test_X = [f.path for f in self.input() if f'_X_test_fold_{fold}.csv' in f.path][0]
			input_train_Y = [f.path for f in self.input() if f'_Y_train_fold_{fold}.csv' in f.path][0]
			input_test_Y = [f.path for f in self.input() if f'_Y_test_fold_{fold}.csv' in f.path][0]
			df_train_X = pd.read_csv(input_train_X, index_col="Datetime", parse_dates=True)
			df_test_X = pd.read_csv(input_test_X, index_col="Datetime", parse_dates=True)
			df_train_Y = pd.read_csv(input_train_Y, index_col="Datetime", parse_dates=True)
			df_test_Y = pd.read_csv(input_test_Y, index_col="Datetime", parse_dates=True)
			df_train = df_train_X.copy()
			df_train[target_column] = df_train_Y[target_column]
			correlation_matrix = df_train.corr().loc[target_column]
			weak_corr_features = [col for col in correlation_matrix.index if -0.15 < correlation_matrix[col] < 0.15 and col != target_column]
			for _ in range(test_number):
				if len(weak_corr_features) < 2:
					break
				x, y = np.random.choice(weak_corr_features, 2, replace=False)
				z = f"{x}_per_{y}"
				df_train[z] = df_train_X[x] / (df_train_X[y] + 1e-6)
				corr = df_train[[z, target_column]].corr().iloc[0, 1]
				if abs(corr) > 0.15:
					print(f"{z} has correlation {corr:.3f} with {target_column}. Feature kept.")
					df_train_X[z] = df_train[z]
					df_test_X[z] = df_test_X[x] / (df_test_X[y] + 1e-6)
				else:
					print(f"{z} correlation {corr:.3f} is too low. Feature discarded.")
			df_train_X.to_csv(output_files[f'X_train_fold_{fold}'].path, index=True)
			df_test_X.to_csv(output_files[f'X_test_fold_{fold}'].path, index=True)
			df_train_Y.to_csv(output_files[f'Y_train_fold_{fold}'].path, index=True)
			df_test_Y.to_csv(output_files[f'Y_test_fold_{fold}'].path, index=True)

class transformStandardScaler(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]

	def output(self):
		output_files = {}
		for input_path in self.input_file:			
			filename = os.path.basename(input_path)
			if '_X_train_fold_' in filename:
				fold = filename.split('_X_train_fold_')[-1].split('.')[0]
				output_files[f'X_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_X_test_fold_' in filename:
				fold = filename.split('_X_test_fold_')[-1].split('.')[0]
				output_files[f'X_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_Y_train_fold_' in filename:
				fold = filename.split('_Y_train_fold_')[-1].split('.')[0]
				output_files[f'Y_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_Y_test_fold_' in filename:
				fold = filename.split('_Y_test_fold_')[-1].split('.')[0]
				output_files[f'Y_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
		return output_files	
	def run(self):
		print("Inside standard scaler ")
		output_files = self.output()
		for fold in range(1, 6):
			input_train_X = [f.path for f in self.input() if f'_X_train_fold_{fold}.csv' in f.path][0]
			input_test_X = [f.path for f in self.input() if f'_X_test_fold_{fold}.csv' in f.path][0]
			input_train_Y = [f.path for f in self.input() if f'_Y_train_fold_{fold}.csv' in f.path][0]
			input_test_Y = [f.path for f in self.input() if f'_Y_test_fold_{fold}.csv' in f.path][0]
			df_train_X = pd.read_csv(input_train_X, index_col="Datetime", parse_dates=True)
			df_test_X = pd.read_csv(input_test_X, index_col="Datetime", parse_dates=True)
			df_train_Y = pd.read_csv(input_train_Y, index_col="Datetime", parse_dates=True)
			df_test_Y = pd.read_csv(input_test_Y, index_col="Datetime", parse_dates=True)
			numeric_train = df_train_X.select_dtypes(include=[np.number])
			numeric_test = df_test_X.select_dtypes(include=[np.number])
			scaler =  StandardScaler()		
			scaler.fit(numeric_train)
			scaled_train = pd.DataFrame(scaler.transform(numeric_train), columns=numeric_train.columns, index=df_train_X.index)
			scaled_test = pd.DataFrame(scaler.transform(numeric_test), columns=numeric_test.columns, index=df_test_X.index)
			non_numeric_train = df_train_X.select_dtypes(exclude=[np.number])
			non_numeric_test = df_test_X.select_dtypes(exclude=[np.number])
			df_train_X_final = pd.concat([scaled_train, non_numeric_train], axis=1)
			df_test_X_final = pd.concat([scaled_test, non_numeric_test], axis=1)
			df_train_X_final.to_csv(output_files[f'X_train_fold_{fold}'].path, index=True)
			df_test_X_final.to_csv(output_files[f'X_test_fold_{fold}'].path, index=True)
			df_train_Y.to_csv(output_files[f'Y_train_fold_{fold}'].path, index=True)
			df_test_Y.to_csv(output_files[f'Y_test_fold_{fold}'].path, index=True)
		
class doPca(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		output_files = {}
		for input_path in self.input_file:
			filename = os.path.basename(input_path)
			if '_X_train_fold_' in filename:
				fold = filename.split('_X_train_fold_')[-1].split('.')[0]
				output_files[f'X_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_X_test_fold_' in filename:
				fold = filename.split('_X_test_fold_')[-1].split('.')[0]
				output_files[f'X_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_Y_train_fold_' in filename:
				fold = filename.split('_Y_train_fold_')[-1].split('.')[0]
				output_files[f'Y_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_Y_test_fold_' in filename:
				fold = filename.split('_Y_test_fold_')[-1].split('.')[0]
				output_files[f'Y_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
		return output_files
	def run(self):
		target_column = self.params['target_column']
		n_components = self.params['n_components']
		output_files = self.output()
		for fold in range(1, 6):
			input_train_X = [f.path for f in self.input() if f'_X_train_fold_{fold}.csv' in f.path][0]
			input_test_X = [f.path for f in self.input() if f'_X_test_fold_{fold}.csv' in f.path][0]
			input_train_Y = [f.path for f in self.input() if f'_Y_train_fold_{fold}.csv' in f.path][0]
			input_test_Y = [f.path for f in self.input() if f'_Y_test_fold_{fold}.csv' in f.path][0]
			df_train_X = pd.read_csv(input_train_X, index_col="Datetime", parse_dates=True)
			df_test_X = pd.read_csv(input_test_X, index_col="Datetime", parse_dates=True)
			df_train_Y = pd.read_csv(input_train_Y, index_col="Datetime", parse_dates=True)
			df_test_Y = pd.read_csv(input_test_Y, index_col="Datetime", parse_dates=True)
			numeric_train = df_train_X.select_dtypes(include=[np.number])
			numeric_test = df_test_X.select_dtypes(include=[np.number])
			pca = PCA(n_components=n_components)
			pca_train = pca.fit_transform(numeric_train)
			pca_test = pca.transform(numeric_test)
			pca_column_names = [f'PC{i+1}' for i in range(n_components)]
			pca_train_df = pd.DataFrame(pca_train, columns=pca_column_names, index=df_train_X.index)
			pca_test_df = pd.DataFrame(pca_test, columns=pca_column_names, index=df_test_X.index)
			pca_train_df.to_csv(output_files[f'X_train_fold_{fold}'].path, index=True)
			pca_test_df.to_csv(output_files[f'X_test_fold_{fold}'].path, index=True)
			df_train_Y.to_csv(output_files[f'Y_train_fold_{fold}'].path, index=True)
			df_test_Y.to_csv(output_files[f'Y_test_fold_{fold}'].path, index=True)
	
class fillMissingValuesImputer(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		output_files = {}
		for input_path in self.input_file:
			filename = os.path.basename(input_path)
			if '_X_train_fold_' in filename:
				fold = filename.split('_X_train_fold_')[-1].split('.')[0]
				output_files[f'X_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_X_test_fold_' in filename:
				fold = filename.split('_X_test_fold_')[-1].split('.')[0]
				output_files[f'X_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_Y_train_fold_' in filename:
				fold = filename.split('_Y_train_fold_')[-1].split('.')[0]
				output_files[f'Y_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_Y_test_fold_' in filename:
				fold = filename.split('_Y_test_fold_')[-1].split('.')[0]
				output_files[f'Y_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
		return output_files	
	def run(self):
		strategy = self.params['strategy']
		print("Inside imputer")
		output_files = self.output()
		for fold in range(1, 6):
			input_train_X = [f.path for f in self.input() if f'_X_train_fold_{fold}.csv' in f.path][0]
			input_test_X = [f.path for f in self.input() if f'_X_test_fold_{fold}.csv' in f.path][0]
			input_train_Y = [f.path for f in self.input() if f'_Y_train_fold_{fold}.csv' in f.path][0]
			input_test_Y = [f.path for f in self.input() if f'_Y_test_fold_{fold}.csv' in f.path][0]
			df_train_X = pd.read_csv(input_train_X, index_col="Datetime", parse_dates=True)
			df_test_X = pd.read_csv(input_test_X, index_col="Datetime", parse_dates=True)
			df_train_Y = pd.read_csv(input_train_Y, index_col="Datetime", parse_dates=True)
			df_test_Y = pd.read_csv(input_test_Y, index_col="Datetime", parse_dates=True)
			numeric_train = df_train_X.select_dtypes(include=[np.number])
			numeric_test = df_test_X.select_dtypes(include=[np.number])
			imputer = SimpleImputer(strategy=strategy)
			imputer.fit(numeric_train)
			imputed_train = pd.DataFrame(imputer.transform(numeric_train), columns=numeric_train.columns, index=df_train_X.index)
			imputed_test = pd.DataFrame(imputer.transform(numeric_test), columns=numeric_test.columns, index=df_test_X.index)
			non_numeric_train = df_train_X.select_dtypes(exclude=[np.number])
			non_numeric_test = df_test_X.select_dtypes(exclude=[np.number])
			df_train_X_final = pd.concat([imputed_train, non_numeric_train], axis=1)
			df_test_X_final = pd.concat([imputed_test, non_numeric_test], axis=1)
			df_train_X_final.to_csv(output_files[f'X_train_fold_{fold}'].path, index=True)
			df_test_X_final.to_csv(output_files[f'X_test_fold_{fold}'].path, index=True)
			df_train_Y.to_csv(output_files[f'Y_train_fold_{fold}'].path, index=True)
			df_test_Y.to_csv(output_files[f'Y_test_fold_{fold}'].path, index=True)
"""	
class divideDatasetIntoModes(OneToOneTask):
	def requires(self):
		return checkDataset(input_file=self.input_file, output_file=self.output_file, params=self.params)

	def output(self):
			output_files = {}
			file_base = os.path.splitext(os.path.basename(self.input_file[0]))[0]
			output_files[f'heating_mode'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_heating_mode.csv"))
			output_files[f'cooling_mode'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_cooling_mode.csv"))
			return output_files
	def run(self):
		print("Inside dividing dataset")
		df = self.convert_csv_to_df()
		cooling_mode = df[(df['AHU: Outdoor Air Temperature'] > df['AHU: Supply Air Temperature']) & 
                          (df['AHU: Outdoor Air Temperature'] > df['AHU: Supply Air Temperature Set Point'])]
		heating_mode = df[(df['AHU: Outdoor Air Temperature'] < df['AHU: Supply Air Temperature']) & 
                          (df['AHU: Outdoor Air Temperature'] < df['AHU: Supply Air Temperature Set Point'])]

		output_files = self.output()
		cooling_mode.to_csv(output_files[f'cooling_mode'].path, index=True)
		heating_mode.to_csv(output_files[f'heating_mode'].path, index=True)
"""




class doOneHotEncoding(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]

	def output(self):
		output_files = {}
		for input_path in self.input_file:
			filename = os.path.basename(input_path)
			if '_X_train_fold_' in filename:
				fold = filename.split('_X_train_fold_')[-1].split('.')[0]
				output_files[f'X_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_X_test_fold_' in filename:
				fold = filename.split('_X_test_fold_')[-1].split('.')[0]
				output_files[f'X_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_Y_train_fold_' in filename:
				fold = filename.split('_Y_train_fold_')[-1].split('.')[0]
				output_files[f'Y_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
			elif '_Y_test_fold_' in filename:
				fold = filename.split('_Y_test_fold_')[-1].split('.')[0]
				output_files[f'Y_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], filename))
		return output_files	
	def run(self):
		output_files = self.output()
		for fold in range(1, 6):
			input_train_X = [f.path for f in self.input() if f'_X_train_fold_{fold}.csv' in f.path][0]
			input_test_X = [f.path for f in self.input() if f'_X_test_fold_{fold}.csv' in f.path][0]
			input_train_Y = [f.path for f in self.input() if f'_Y_train_fold_{fold}.csv' in f.path][0]
			input_test_Y = [f.path for f in self.input() if f'_Y_test_fold_{fold}.csv' in f.path][0]
			df_train_X = pd.read_csv(input_train_X, index_col="Datetime", parse_dates=True)
			df_test_X = pd.read_csv(input_test_X, index_col="Datetime", parse_dates=True)
			df_train_Y = pd.read_csv(input_train_Y, index_col="Datetime", parse_dates=True)
			df_test_Y = pd.read_csv(input_test_Y, index_col="Datetime", parse_dates=True)
			for df in [df_train_X, df_test_X]:
				df['year'] = df.index.year
				df['month'] = df.index.month
				df['day'] = df.index.day
				df['hour'] = df.index.hour

			datetime_columns = ['year', 'month', 'day', 'hour']
			encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
			
			
			encoded_train = encoder.fit_transform(df_train_X[datetime_columns])
			encoded_test = encoder.transform(df_test_X[datetime_columns])

			encoded_column_names = encoder.get_feature_names_out(datetime_columns)

			
			encoded_train_df = pd.DataFrame(encoded_train, columns=encoded_column_names, index=df_train_X.index)
			encoded_test_df = pd.DataFrame(encoded_test, columns=encoded_column_names, index=df_test_X.index)

			df_train_X_final = pd.concat([df_train_X.drop(columns=datetime_columns), encoded_train_df], axis=1)
			df_test_X_final = pd.concat([df_test_X.drop(columns=datetime_columns), encoded_test_df], axis=1)

			
			df_train_X_final.to_csv(output_files[f'X_train_fold_{fold}'].path, index=True)
			df_test_X_final.to_csv(output_files[f'X_test_fold_{fold}'].path, index=True)
			df_train_Y.to_csv(output_files[f'Y_train_fold_{fold}'].path, index=True)
			df_test_Y.to_csv(output_files[f'Y_test_fold_{fold}'].path, index=True)
class PreprocessingPipeline(luigi.WrapperTask):	
	input_dir = luigi.Parameter() 
	output_dir = luigi.Parameter()
	config = luigi.Parameter() 
	tasks = {
		#'check_dataset': checkDataset,
		'calculate_history': calculateHistory,
		#'calculate_rolling_window': calculateRollingWindow,
		'calculate_sine_cosaine': calculateSineCosaineHour,
		'attribute_adder': attributeAddr,
		'standard_scaler': transformStandardScaler,
		'imputer': fillMissingValuesImputer,
		'pca' : doPca,
		'one_hot_encoding' : doOneHotEncoding,
		'stratified_split': stratifiedSplit,
		'time_series_split' : timeSeriesSplit,
		#'divide_dataset_into_modes': divideDatasetIntoModes


	}

	task_mapping = {
		#'check_dataset':['csv', 'csv'],
		'calculate_history': ['csv', 'csv'],
		#'calculate_rolling_window': ['csv', 'csv'],
		'calculate_sine_cosaine': ['csv', 'csv'],
		'attribute_adder':  ['csv', 'csv'],
		'standard_scaler': ['csv', 'csv'],
		'imputer': ['csv', 'csv'],
		'pca': ['csv', 'csv'],
		'one_hot_encoding' : ['csv', 'csv'],
		'stratified_split': ['csv', 'csv'],
		'time_series_split': ['csv', 'csv'],
		#'divide_dataset_into_modes': ['csv', 'csv']
	
	}
	
	def requires(self):
		config = self.parse_config()
		pipeline_config = config['pipeline']

		for task in pipeline_config:
			
			task_type = task['task']
			unique_id = task['id']
			input_id = task['input_id']
			params = task['parameters']
			
			
			input_dir = self.input_dir if input_id == 'input' else os.path.join(self.output_dir, input_id)
			output_dir = os.path.join(self.output_dir, unique_id)
			
			
			input_tree = self.get_directory_tree(input_dir)
			for subdir in input_tree:
				os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

			
			input_format, output_format = self.get_io_format(task)
			input_files = self.get_files(input_dir, input_format)
			output_files = []
			for file in input_files:
				file = os.path.relpath(file, input_dir)
				directory_name = os.path.dirname(file)
				
				output_files.append(os.path.join(output_dir,directory_name))
				
			mapping = list(zip(input_files, output_files))
		

			
			pending_tasks = []
			if len(input_files) > 1:
				pending_tasks.append(
                self.tasks[task_type](
                    input_file=input_files,  
                    output_file=output_files,  
                    params=params
                )
            )
			else:
				for input_file, output_file in mapping:
					print("OUTPUT_FILE", output_file)
                    
					pending_tasks.append(
                    self.tasks[task_type](
                        input_file=[input_file],  
                        output_file=[output_file],  
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
				filename, ext = os.path.splitext(file)
				ext = ext.strip(".")
				if ext == file_format:
					file = os.path.join(root, file)
					target_files.append(file)
		return target_files

	def get_io_format(self, task):
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

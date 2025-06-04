import os 
import argparse
import yaml
import luigi 
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_docs as tfdocs
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import tensorflow_docs.modeling


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
		important_feature = self.params['important_feature']
		target_column = self.params['target_column']
		df = pd.read_csv(self.input()[0].path, index_col="Datetime", parse_dates=True)
		if important_feature not in df.columns:
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



class timeBasedSplit(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		output_files = {}
		file_base = os.path.splitext(os.path.basename(self.input_file[0]))[0]
		for fold in range(1, 6):  # 5 folds
			output_files[f'X_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_X_train_fold_{fold}.csv"))
			output_files[f'Y_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_Y_train_fold_{fold}.csv"))
			output_files[f'X_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_X_test_fold_{fold}.csv"))
			output_files[f'Y_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_Y_test_fold_{fold}.csv"))
		return output_files
	def run(self):
		target_column = self.params['target_column']
		df = pd.read_csv(self.input()[0].path, index_col="Datetime", parse_dates=True)

        # Sort the dataset by time
		df = df.sort_index()
		
		chunk_size = len(df) // 5
		indices = np.arange(len(df))
		chunks = [indices[i * chunk_size: (i + 1) * chunk_size] for i in range(4)]
		chunks.append(indices[4 * chunk_size:])  # Add the remaining rows to the last chunk
		for fold in range(1, 6):
			test_indices = chunks[fold - 1]
			train_indices = np.concatenate([chunks[i] for i in range(5) if i != (fold - 1)])
			train_dates = df.index[train_indices]
			test_dates = df.index[test_indices]
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
        for fold in range(1, 6):  # 5 folds
            output_files[f'X_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_X_train_fold_{fold}.csv"))
            output_files[f'Y_train_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_Y_train_fold_{fold}.csv"))
            output_files[f'X_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_X_test_fold_{fold}.csv"))
            output_files[f'Y_test_fold_{fold}'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_Y_test_fold_{fold}.csv"))
        return output_files

    def run(self):
        target_column = self.params['target_column']
        df = pd.read_csv(self.input()[0].path, index_col="Datetime", parse_dates=True)

        # Sort the dataset by time
        df = df.sort_index()

        # TimeSeriesSplit for continuous time-based splits
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
            
            # Save the train and test sets for the current fold
            X_train.to_csv(output_files[f'X_train_fold_{fold}'].path, index=True)
            Y_train.to_csv(output_files[f'Y_train_fold_{fold}'].path, index=True)
            X_test.to_csv(output_files[f'X_test_fold_{fold}'].path, index=True)
            Y_test.to_csv(output_files[f'Y_test_fold_{fold}'].path, index=True)




class lstm(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		expected =self.params['expected_file_name']
		predicted = self.params['predicted_file_name']
		file_format = self.params['file_format']
		output_files = {}
		for fold in range(1, 6):  
			output_files[f'predicted_fold_{fold}'] = luigi.LocalTarget(
                os.path.join(self.output_file[0], f"{predicted}_fold_{fold}.{file_format}")
            )
			output_files[f'expected_fold_{fold}'] = luigi.LocalTarget(
                os.path.join(self.output_file[0], f"{expected}_fold_{fold}.{file_format}")
            )
		return output_files

        
	def run(self):
		expected =self.params['expected_file_name']
		predicted = self.params['predicted_file_name']
		file_format = self.params['file_format']
		print("In LSTM input")
		for fold in range(1, 6):
			dataframes = {'X_train': None, 'Y_train': None, 'X_test': None, 'Y_test': None}
			for input_target in self.input():
				file_base = os.path.splitext(os.path.basename(input_target.path))[0]
				for key in dataframes.keys():
					if f"{key}_fold_{fold}" in file_base:
						dataframes[key] = pd.read_csv(input_target.path, index_col="Datetime", parse_dates=True)
			print("shape")
			print(dataframes['X_train'].shape)
			dimension1 = 24
			dimension2 = dataframes['X_train'].shape[1]
			X_train, Y_train = self.create_lstm_input(dataframes['X_train'].values, dataframes['Y_train'].values)
			X_test, Y_test, index = self.reshape(dataframes['X_test'].values, dataframes['Y_test'].values,  dataframes['Y_test'].index)
			print("Reshaped X_train shape:", X_train.shape)
			print("Reshaped Y_train shape:", Y_train.shape)
			print("Reshaped X_test shape:", X_test.shape)
			print("Reshaped Y_test shape:", Y_test.shape)
			print(index)
			
	
			model = self.build_lstm_model(dimension1, dimension2)
			model.fit(X_train, Y_train,
                   epochs=10,
                  batch_size=512, #
                  callbacks=[tfdocs.modeling.EpochDots()])
			predictions = model.predict(X_test)
			predicted_data = pd.Series(predictions.ravel(), index=index)
			predicted_df = pd.DataFrame({'Predicted': predicted_data, 'Probability' :predicted_data }, index= index)
			Y_test_df = pd.DataFrame(Y_test, columns=['Actual'], index= index)
			output_files = self.output()
			predicted_df.to_csv(output_files[f'{predicted}_fold_{fold}'].path, index=True)
			Y_test_df.to_csv(output_files[f'{expected}_fold_{fold}'].path, index=True)
	
	def build_lstm_model(self, dimension1, dimension2):
		lstm1 = tf.keras.layers.LSTM(
            input_shape=(dimension1, dimension2), units=400, activation='tanh',
            recurrent_activation='sigmoid', use_bias=True, dropout=0.2, return_sequences=True
        )
		lstm2 = tf.keras.layers.LSTM(
            units=400, activation='tanh',
            recurrent_activation='sigmoid', use_bias=True, dropout=0.2, return_sequences=False
        )
		mlp1 = tf.keras.layers.Dense(15, activation="sigmoid")
		mlp2 = tf.keras.layers.Dense(15, activation="sigmoid")
		mlp3 = tf.keras.layers.Dense(1, activation="sigmoid")
		dropout = tf.keras.layers.Dropout(0.2, noise_shape=None) #
		inputs = keras.Input(shape=( dimension1,dimension2))
		x0=lstm1(inputs)
		x1 = lstm2(x0)
		x2 = mlp1(x1)
		x3=mlp2(x2)
		x4=mlp3(x3)
		model = keras.Model(inputs=inputs, outputs=x4, name="LSTM_5Fold")
		CE = tf.keras.losses.BinaryCrossentropy(
    	from_logits=False, label_smoothing=0, reduction='sum_over_batch_size',
   		name='binary_crossentropy')
		model.compile( optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
                loss=CE,
                metrics=[CE])
		return model
	def create_lstm_input(self, features, target, timesteps=24):
		X = [features[i:i + timesteps, :] for i in range(len(features) - timesteps)]
		y = [target[i + timesteps] for i in range(len(target) - timesteps)] if target is not None else None
		return np.array(X), np.array(y)

		
	def reshape(self,features,target, index,timesteps=24):
		num_samples = len(features) - timesteps
		X_test = np.array([features[i:i + timesteps, :] for i in range(num_samples)])
		Y_test = target[timesteps:]
		idx_test = index[timesteps:] if index is not None else None  # Align index with Y_test
		return X_test, Y_test, idx_test

class randomforest(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		expected =self.params['expected_file_name']
		predicted = self.params['predicted_file_name']
		file_format = self.params['file_format']
		output_files = {}
		for fold in range(1, 6):  
			output_files[f'predicted_fold_{fold}'] = luigi.LocalTarget(
                os.path.join(self.output_file[0], f"{predicted}_fold_{fold}.{file_format}")
            )
			output_files[f'expected_fold_{fold}'] = luigi.LocalTarget(
                os.path.join(self.output_file[0], f"{expected}_fold_{fold}.{file_format}")
            )
		return output_files

        
	def run(self):
		expected =self.params['expected_file_name']
		predicted = self.params['predicted_file_name']
		file_format = self.params['file_format']
		n_estimators = self.params['n_estimators'] 
		max_depth = self.params['max_depth'] 
		min_samples_split = self.params['min_samples_split'] 
		min_samples_leaf = self.params['min_samples_leaf'] 
		bootstrap = self.params['bootstrap'] 
		max_features = self.params['max_features']  
		print("In random forest")
		for fold in range(1, 6):
			dataframes = {'X_train': None, 'Y_train': None, 'X_test': None, 'Y_test': None}
			for input_target in self.input():
				file_base = os.path.splitext(os.path.basename(input_target.path))[0]
				if f'_fold_{fold}' in file_base:
					for key in dataframes.keys():
						if key in file_base:
							dataframes[key] = pd.read_csv(input_target.path, index_col="Datetime", parse_dates=True)
			if any(v is None for v in dataframes.values()):
				raise ValueError(f"Data missing for fold {fold} in one of the required sets: {dataframes.keys()}")
			rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            random_state=42)
			rf_model.fit(dataframes['X_train'], dataframes['Y_train'].values.ravel())
			prediction = rf_model.predict(dataframes['X_test'])
			probabilities = rf_model.predict_proba(dataframes['X_test'])
			print(probabilities.shape)
			if probabilities.ndim == 2 and probabilities.shape[1] == 2:
				probabilities = probabilities[:, 1]
			predicted_data = pd.Series(prediction, index=dataframes['Y_test'].index)
			predicted_df = pd.DataFrame({'Predicted': predicted_data,'Probability': probabilities.ravel()})

            # Save predicted and expected output for the current fold
			output_files = self.output()
			predicted_df.to_csv(output_files[f'{predicted}_fold_{fold}'].path, index=True)
			dataframes['Y_test'].to_csv(output_files[f'{expected}_fold_{fold}'].path, index=True)	
				
		
class FullyConnectedNN(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		expected =self.params['expected_file_name']
		predicted = self.params['predicted_file_name']
		file_format = self.params['file_format']
		output_files = {}
		for fold in range(1, 6):  
			output_files[f'predicted_fold_{fold}'] = luigi.LocalTarget(
                os.path.join(self.output_file[0], f"{predicted}_fold_{fold}.{file_format}")
            )
			output_files[f'expected_fold_{fold}'] = luigi.LocalTarget(
                os.path.join(self.output_file[0], f"{expected}_fold_{fold}.{file_format}")
            )
		return output_files
	def build_fcnn_model(self, input_dim, hidden_neurons=30, output_dim=1):
		inputs = layers.Input(shape=(input_dim,))
		hidden = layers.Dense(hidden_neurons, activation='relu')(inputs)
		outputs = layers.Dense(output_dim, activation='sigmoid')(hidden)
		model = models.Model(inputs, outputs, name="FCNN")
		model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
		return model
	def run(self):
		expected = self.params['expected_file_name']
		predicted = self.params['predicted_file_name']
		file_format = self.params['file_format']
		hidden_neurons = self.params['hidden_neurons']
		for fold in range(1, 6):
			dataframes = {'X_train': None, 'Y_train': None, 'X_test': None, 'Y_test': None}
			for input_target in self.input():
				file_base = os.path.splitext(os.path.basename(input_target.path))[0]
				if f'_fold_{fold}' in file_base:
					for key in dataframes.keys():
						if key in file_base:
							dataframes[key] = pd.read_csv(input_target.path, index_col="Datetime", parse_dates=True)
			if any(v is None for v in dataframes.values()):
				raise ValueError(f"Data missing for fold {fold} in one of the required sets: {dataframes.keys()}")
			input_dim = dataframes['X_train'].shape[1]
			model = self.build_fcnn_model(input_dim=input_dim, hidden_neurons=hidden_neurons)
			model.fit(
                dataframes['X_train'].values,
                dataframes['Y_train'].values.ravel(),
                epochs=10,
                batch_size=512,
                verbose=1
            )
			predictions = model.predict(dataframes['X_test'].values)
			predicted_df = pd.DataFrame({
                'Predicted': predictions.ravel(),
                'Probability': predictions.ravel()
            }, index=dataframes['Y_test'].index)
			output_files = self.output()
			predicted_df.to_csv(output_files[f'{predicted}_fold_{fold}'].path, index=True)
			dataframes['Y_test'].to_csv(output_files[f'{expected}_fold_{fold}'].path, index=True)

		


class arima(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		expected =self.params['expected_file_name']
		predicted = self.params['predicted_file_name']
		file_format = self.params['file_format']
		output_files = {}
		for fold in range(1, 6):
			output_files[f'{predicted}_fold_{fold}'] = luigi.LocalTarget(
                os.path.join(self.output_file[0], f"{predicted}_fold_{fold}.{file_format}")
            )
			output_files[f'{expected}_fold_{fold}'] = luigi.LocalTarget(
                os.path.join(self.output_file[0], f"{expected}_fold_{fold}.{file_format}")
            )
		return output_files

	def run(self):
		expected =self.params['expected_file_name']
		predicted = self.params['predicted_file_name']
		file_format = self.params['file_format']
		number_of_time_lags =self.params['number_of_time_lags']
		degree_of_differencing = self.params['degree_of_differencing']
		moving_average = self.params['moving_average']
		print("In arima input")
		for fold in range(1, 6):
			dataframes = {'X_train': None, 'Y_train': None, 'X_test': None, 'Y_test': None}
			for input_target in self.input():
				file_base = os.path.splitext(os.path.basename(input_target.path))[0]
				if f'_fold_{fold}' in file_base:
					for key in dataframes.keys():
						if key in file_base:
							dataframes[key] = pd.read_csv(input_target.path, index_col="Datetime", parse_dates=True)
			if any(v is None for v in dataframes.values()):
				raise ValueError(f"Data missing for fold {fold} in one of the required sets: {dataframes.keys()}")
			dataframes['Y_train'].index = pd.DatetimeIndex(dataframes['Y_train'].index).to_period('min')
			dataframes['X_train'].index = pd.DatetimeIndex(dataframes['X_train'].index).to_period('min')
			dataframes['X_test'].index = pd.DatetimeIndex(dataframes['X_test'].index).to_period('min')

			A_Model = ARIMA(dataframes['Y_train'], exog=dataframes['X_train'], order=(number_of_time_lags, degree_of_differencing, moving_average))
			Arima_Model = A_Model.fit()
			prediction = Arima_Model.predict(
                start=len(dataframes['Y_train']),
                end=len(dataframes['Y_train']) + len(dataframes['Y_test']) - 1,
                exog=dataframes['X_test'],
                typ='levels'
            )
			predicted_data = pd.Series(prediction.values, index=dataframes['Y_test'].index)
			predicted_df = pd.DataFrame(predicted_data, columns=['Predicted'])

            # Save predicted and expected output for the current fold
			output_files = self.output()
			predicted_df.to_csv(output_files[f'{predicted}_fold_{fold}'].path, index=True)
			dataframes['Y_test'].to_csv(output_files[f'{expected}_fold_{fold}'].path, index=True)		
class evaluation(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		eval = self.params['eval_file_name']
		file_format = self.params['file_format']
		
		output_files = {
            eval: luigi.LocalTarget(os.path.join(self.output_file[0], f"{eval}.{file_format}"))           
        }
        
		return output_files
	def run(self):
		eval = self.params['eval_file_name']
		expected =self.params['expected_file_name']
		predicted = self.params['predicted_file_name']
		continuous = self.params['continuous']
		dataframes = {expected: None,predicted: None}
		threshold = self.params['threshold']
		all_metrics = []
		for fold in range(1, 6):
			dataframes = {expected: None, predicted: None}
			for input_target in self.input():
				file_base = os.path.splitext(os.path.basename(input_target.path))[0]
				if f'_fold_{fold}' in file_base:
					for key in dataframes.keys():
						if key in file_base:
							dataframes[key] = pd.read_csv(input_target.path, index_col="Datetime", parse_dates=True)
			dataframes[predicted] = (dataframes[predicted] >= threshold).astype(int)
			values = pd.DataFrame(index=(dataframes[expected].index))
			values['Actual'] = dataframes[expected]
			values['Predicted'] = dataframes[predicted]['Predicted']

			values['TP'] = ((values['Actual'] == 1) & (values['Predicted'] == 1)).astype(int)
			values['FP'] = ((values['Actual'] == 0) & (values['Predicted'] == 1)).astype(int)
			values['FN'] = ((values['Actual'] == 1) & (values['Predicted'] == 0)).astype(int)
			values['TN'] = ((values['Actual'] == 0) & (values['Predicted'] == 0)).astype(int)
			tp, fp, fn, tn = values['TP'].sum(), values['FP'].sum(), values['FN'].sum(), values['TN'].sum()
			accuracy = (tp + tn) / (tp + fp + fn + tn)
			precision = tp / (tp + fp) if (tp + fp) > 0 else 0
			recall = tp / (tp + fn) if (tp + fn) > 0 else 0
			f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
			specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
			fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
			fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
			metrics = {
                'Fold': fold,
                'True Positive': tp,
				'False Positive': fp,
                'True Negative': tn,
                'False Negative': fn,
                
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1_score,
                'Specificity': specificity,
                'False Positive Rate': fpr,
				'False Negative Rate' : fnr
            }
			if continuous == 0:
				actuals = values['Actual'].values
				probs = dataframes[predicted]['Probability'].values.astype(float)
				epsilon = 1e-15
				probs = np.clip(probs, epsilon, 1 - epsilon)
				cross_entropy = -np.mean(actuals * np.log(probs) + (1 - actuals) * np.log(1 - probs))
				metrics['Cross-Entropy'] = cross_entropy
			all_metrics.append(metrics)
		eval_df = pd.DataFrame(all_metrics)
		eval_df.to_csv(self.output()[f'{eval}'].path, index = False)
	
class divideDatasetIntoModes(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]

	def output(self):
			output_files = {}
			file_base = os.path.splitext(os.path.basename(self.input_file[0]))[0]
			output_files[f'heating_mode'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_heating_mode.csv"))
			output_files[f'cooling_mode'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_cooling_mode.csv"))
			return output_files
	def run(self):
		print("Inside dividing dataset")
		df = pd.read_csv(self.input()[0].path, index_col="Datetime", parse_dates=True)
		cooling_mode = df[(df['AHU: Outdoor Air Temperature'] >= df['AHU: Supply Air Temperature'])]
		heating_mode = df[(df['AHU: Outdoor Air Temperature'] < df['AHU: Supply Air Temperature'])]

		output_files = self.output()
		cooling_mode.to_csv(output_files[f'cooling_mode'].path, index=True)
		heating_mode.to_csv(output_files[f'heating_mode'].path, index=True)




	
		
class ClassificationPipeline(luigi.WrapperTask):	
	input_dir = luigi.Parameter() 
	output_dir = luigi.Parameter()
	config = luigi.Parameter() 
	tasks = {
		'stratified_split': stratifiedSplit,
		'arima': arima,
		'evaluation': evaluation,
		'time_based_split' : timeBasedSplit,
		'time_series_split' : timeSeriesSplit,
		'lstm' : lstm,
		'randomforest' : randomforest,
		'fullyConnectedNN' : FullyConnectedNN,
		'divide_dataset_into_modes': divideDatasetIntoModes

	}
	task_mapping = {
		'stratified_split': ['csv', 'csv'],
		'arima' : ['csv', 'csv'],
		'evaluation': ['csv', 'csv'],
		'time_based_split': ['csv', 'csv'],
		'time_series_split' : ['csv', 'csv'],
		'lstm': ['csv', 'csv'],
		'randomforest'  : ['csv', 'csv'],
		'fullyConnectedNN': ['csv', 'csv'],
		'divide_dataset_into_modes' : ['csv', 'csv']
		
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
				directory_name = os.path.dirname(file)
				
				output_files.append(os.path.join(output_dir,directory_name))
				
			mapping = list(zip(input_files, output_files))
		

			# Instantiating Tasks
			pending_tasks = []
			if len(input_files) > 1:
				pending_tasks.append(
                self.tasks[task_type](
                    input_file=input_files,  # Pass all input files
                    output_file=output_files,  # Pass all output files
                    params=params
                )
            )
			else:
				for input_file, output_file in mapping:
					print("OUTPUT_FILE", output_file)
                    
					pending_tasks.append(
                    self.tasks[task_type](
                        input_file=[input_file],  # Wrap in a list
                        output_file=[output_file],  # Wrap in a list
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
		[ClassificationPipeline(
			input_dir=args.input, 
			output_dir=args.output, 
			config=args.config
		)], 
		scheduler_host='localhost', 
		scheduler_port=8082
	)

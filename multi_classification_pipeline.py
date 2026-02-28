import os 
import argparse
import yaml
import luigi 
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import json
import tensorflow_docs as tfdocs
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import tensorflow_docs.modeling
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors



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




class lstm(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		expected = self.params['expected_file_name']
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
			output_files[f'model_fold_{fold}'] = luigi.LocalTarget(
                os.path.join(self.output_file[0], f"lstm_fold_{fold}.joblib")
            )
		return output_files
	def run(self):
		expected = self.params['expected_file_name']
		predicted = self.params['predicted_file_name']
		file_format = self.params['file_format']
		print("In LSTM input")
		for fold in range(1, 6):
			dataframes = {'X_train': None, 'Y_train': None, 'X_test': None, 'Y_test': None}
			for input_target in self.input():
				file_base = os.path.splitext(os.path.basename(input_target.path))[0]
				for key in dataframes.keys():
					if f"{key}_fold_{fold}" in file_base:
						dataframes[key] = pd.read_csv(
                            input_target.path,
                            index_col="Datetime",
                            parse_dates=True
                        )
			if any(v is None for v in dataframes.values()):
				raise ValueError(
                    f"Data missing for fold {fold} in one of the required sets: {list(dataframes.keys())}"
                )
			print("shape")
			print(dataframes['X_train'].shape)
			timesteps = 24
			feature_dim = dataframes['X_train'].shape[1]

            
            
			y_train_series = dataframes['Y_train'].iloc[:, 0]
			classes = np.sort(y_train_series.unique())
			class_to_index = {cls: i for i, cls in enumerate(classes)}
			index_to_class = {i: cls for cls, i in class_to_index.items()}
			y_train_int_full = y_train_series.map(class_to_index).values
			if np.any(pd.isna(y_train_int_full)):
				raise ValueError("Some training labels could not be mapped to class indices.")
			X_train, Y_train_int = self.create_lstm_input(
                dataframes['X_train'].values,
                y_train_int_full,
                timesteps=timesteps
            )
			X_test, Y_test, index = self.reshape(
                dataframes['X_test'].values,
                dataframes['Y_test'].values,
                dataframes['Y_test'].index
            )
			print("Reshaped X_train shape:", X_train.shape)
			print("Reshaped Y_train shape:", Y_train_int.shape)
			print("Reshaped X_test shape:", X_test.shape)
			print("Reshaped Y_test shape:", Y_test.shape)
			print(index)

            
			model = self.build_lstm_model(
                timesteps=timesteps,
                feature_dim=feature_dim,
                num_classes=len(classes)
            )
			model.fit(
                X_train,
                Y_train_int,
                epochs=10,
                batch_size=512,
                verbose=1,
                callbacks=[tfdocs.modeling.EpochDots()]
            )
			proba = model.predict(X_test)  
			pred_indices = np.argmax(proba, axis=1)
			pred_labels = [index_to_class[i] for i in pred_indices]
			proba_cols = {
                f"Prob_{cls}": proba[:, idx]
                for idx, cls in enumerate(classes)
            }
			predicted_series = pd.Series(
                pred_labels,
                index=index,
                name='Predicted'
            )
			proba_df = pd.DataFrame(
                proba_cols,
                index=index
            )
			predicted_df = pd.concat([predicted_series, proba_df], axis=1)

            
			Y_test_df = pd.DataFrame(Y_test, columns=['Actual'], index=index)
			output_files = self.output()
			predicted_df.to_csv(output_files[f'{predicted}_fold_{fold}'].path, index=True)
			Y_test_df.to_csv(output_files[f'{expected}_fold_{fold}'].path, index=True)
			joblib.dump(model, output_files[f'model_fold_{fold}'].path)
			
	
	def build_lstm_model(self, timesteps, feature_dim, num_classes):
        
		lstm1 = tf.keras.layers.LSTM(
            input_shape=(timesteps, feature_dim),
            units=400,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=True,
            dropout=0.2,
            return_sequences=True
        )
		lstm2 = tf.keras.layers.LSTM(
            units=400,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=True,
            dropout=0.2,
            return_sequences=False
        )

        
		mlp1 = tf.keras.layers.Dense(15, activation="sigmoid")
		mlp2 = tf.keras.layers.Dense(15, activation="sigmoid")
        
		mlp3 = tf.keras.layers.Dense(num_classes, activation="softmax")
		dropout = tf.keras.layers.Dropout(0.2)
		inputs = keras.Input(shape=(timesteps, feature_dim))
		x0 = lstm1(inputs)
		x1 = lstm2(x0)
		x2 = mlp1(x1)
		x2 = dropout(x2)
		x3 = mlp2(x2)
		x3 = dropout(x3)
		outputs = mlp3(x3)
		model = keras.Model(inputs=inputs, outputs=outputs, name="LSTM_5Fold_Multiclass")
		CE = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
            name='sparse_categorical_crossentropy'
        )
		model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
            loss=CE,
            metrics=['accuracy']
        )
		return model
	def create_lstm_input(self, features, target, timesteps=24):
		"""
        features: array of shape (N, D)
        target:   array of shape (N,) with integer class indices
        returns:
            X: (N - timesteps, timesteps, D)
            y: (N - timesteps,) corresponding class indices
        """
		X = [features[i:i + timesteps, :] for i in range(len(features) - timesteps)]
		y = [target[i + timesteps] for i in range(len(target) - timesteps)] if target is not None else None
		return np.array(X), np.array(y)
	def reshape(self,features,target, index,timesteps=24):
		num_samples = len(features) - timesteps
		X_test = np.array([features[i:i + timesteps, :] for i in range(num_samples)])
		Y_test = target[timesteps:]
		idx_test = index[timesteps:] if index is not None else None  
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
			output_files[f'model_fold_{fold}'] = luigi.LocalTarget(
                os.path.join(self.output_file[0], f"randomforest_fold_{fold}.joblib")
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
            bootstrap=bootstrap,
            random_state=42)
			rf_model.fit(dataframes['X_train'], dataframes['Y_train'].values.ravel())
			prediction = rf_model.predict(dataframes['X_test'])
			probabilities = rf_model.predict_proba(dataframes['X_test'])
			print("Probabilities shape:", probabilities.shape)
			
			class_labels = rf_model.classes_
			proba_df = pd.DataFrame(
                probabilities,
                index=dataframes['Y_test'].index,
                columns=[f"Prob_{cls}" for cls in class_labels]
            )
			predicted_series = pd.Series(prediction, index=dataframes['Y_test'].index, name='Predicted')
			predicted_df = pd.concat([predicted_series, proba_df], axis=1)

            
			output_files = self.output()
			predicted_df.to_csv(output_files[f'{predicted}_fold_{fold}'].path, index=True)
			dataframes['Y_test'].to_csv(output_files[f'{expected}_fold_{fold}'].path, index=True)
			joblib.dump(rf_model, output_files[f'model_fold_{fold}'].path)
				
		
class FullyConnectedNN(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		expected = self.params['expected_file_name']
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
			output_files[f'model_fold_{fold}'] = luigi.LocalTarget(
                os.path.join(self.output_file[0], f"fullyConnectedNN_fold_{fold}.joblib")
            )
		return output_files
	def build_fcnn_model(self, input_dim, hidden_neurons=30, num_classes=2):
		inputs = layers.Input(shape=(input_dim,))
		hidden = layers.Dense(hidden_neurons, activation='relu')(inputs)
		outputs = layers.Dense(num_classes, activation='softmax')(hidden)
		model = models.Model(inputs, outputs, name="FCNN")
		model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',  
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
							dataframes[key] = pd.read_csv(
                                input_target.path,
                                index_col="Datetime",
                                parse_dates=True
                            )
			if any(v is None for v in dataframes.values()):
				raise ValueError(
                    f"Data missing for fold {fold} in one of the required sets: {list(dataframes.keys())}"
                )
			input_dim = dataframes['X_train'].shape[1]
			y_train_series = dataframes['Y_train'].iloc[:, 0]
			classes = np.sort(y_train_series.unique())
			class_to_index = {cls: i for i, cls in enumerate(classes)}
			index_to_class = {i: cls for cls, i in class_to_index.items()}
			y_train_int = y_train_series.map(class_to_index).values
			if np.any(pd.isna(y_train_int)):
				raise ValueError("Some training labels could not be mapped to class indices.")
			model = self.build_fcnn_model(
                input_dim=input_dim,
                hidden_neurons=hidden_neurons,
                num_classes=len(classes)
            )
			model.fit(
                dataframes['X_train'].values,
                y_train_int,
                epochs=10,
                batch_size=512,
                verbose=1
            )
			proba = model.predict(dataframes['X_test'].values)  
			pred_indices = np.argmax(proba, axis=1)
			pred_labels = [index_to_class[i] for i in pred_indices]
			proba_cols = {
                f"Prob_{cls}": proba[:, idx]
                for idx, cls in enumerate(classes)
            }
			predicted_series = pd.Series(
                pred_labels,
                index=dataframes['Y_test'].index,
                name='Predicted'
            )
			proba_df = pd.DataFrame(
                proba_cols,
                index=dataframes['Y_test'].index
            )
			predicted_df = pd.concat([predicted_series, proba_df], axis=1)
			output_files = self.output()
			predicted_df.to_csv(output_files[f'{predicted}_fold_{fold}'].path, index=True)
			dataframes['Y_test'].to_csv(output_files[f'{expected}_fold_{fold}'].path, index=True)
			joblib.dump(model, output_files[f'model_fold_{fold}'].path)

		



class evaluation(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		eval_name = self.params['eval_file_name']
		file_format = self.params['file_format']
		output_files = {
            
            eval_name: luigi.LocalTarget(
                os.path.join(self.output_file[0], f"{eval_name}.{file_format}")
            ),
            
            f"{eval_name}_per_class": luigi.LocalTarget(
                os.path.join(self.output_file[0], f"{eval_name}_per_class.{file_format}")
            ),
            
            f"{eval_name}_confusion_matrix_csv": luigi.LocalTarget(
                os.path.join(self.output_file[0], f"{eval_name}_confusion_matrix.{file_format}")
            ),
            
            f"{eval_name}_confusion_matrix_plot": luigi.LocalTarget(
                os.path.join(self.output_file[0], f"{eval_name}_confusion_matrix.jpg")
            ),
        }
		return output_files
	def run(self):
		eval_name = self.params['eval_file_name']
		expected = self.params['expected_file_name']
		predicted = self.params['predicted_file_name']
		outputs = self.output()
		
		all_metrics = []         
		y_true_all = []          
		y_pred_all = []
		
		for fold in range(1, 6):
			dataframes = {expected: None, predicted: None}
			
			for input_target in self.input():
				file_base = os.path.splitext(os.path.basename(input_target.path))[0]
				if f"_fold_{fold}" in file_base:
					for key in dataframes.keys():
						if key in file_base:
							dataframes[key] = pd.read_csv(
                                input_target.path,
                                index_col="Datetime",
                                parse_dates=True
                            )
			if any(v is None for v in dataframes.values()):
				raise ValueError(
                    f"Data missing for fold {fold} in one of the required sets: "
                    f"{list(dataframes.keys())}"
                )
			if "Predicted" not in dataframes[predicted].columns:
				raise ValueError(
                    f"Predicted file for fold {fold} must contain a 'Predicted' column "
                    f"with class labels."
                )
			
			y_true = dataframes[expected].iloc[:, 0].values
			y_pred = dataframes[predicted]["Predicted"].values
			
			y_true_all.extend(y_true)
			y_pred_all.extend(y_pred)
			
			class_labels = np.unique(
                np.concatenate([np.unique(y_true), np.unique(y_pred)])
            )
			accuracy = accuracy_score(y_true, y_pred)
			macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
			weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
			metrics = {
                "Fold": fold,
                "Accuracy": accuracy,
                "Macro_F1": macro_f1,
                "Weighted_F1": weighted_f1,
            }
			
			pred_df = dataframes[predicted]
			prob_cols = [c for c in pred_df.columns if c.startswith("Prob_")]
			if prob_cols:
				proba_df = pred_df[prob_cols]
				probs_true = []
				for idx, label in enumerate(y_true):
					col_name = f"Prob_{label}"
					if col_name in proba_df.columns:
						p = float(proba_df.iloc[idx][col_name])
					else:
						p = 1.0 / len(class_labels)
					probs_true.append(p)
				
				probs_true = np.array(probs_true, dtype=float)
				epsilon = 1e-15
				probs_true = np.clip(probs_true, epsilon, 1 - epsilon)
				cross_entropy = -np.mean(np.log(probs_true))
				metrics["CrossEntropy"] = cross_entropy
			
			elif "Probability" in pred_df.columns:
				actuals = y_true.astype(int)
				probs = pred_df["Probability"].values.astype(float)
				epsilon = 1e-15
				probs = np.clip(probs, epsilon, 1 - epsilon)
				cross_entropy = -np.mean(
                    actuals * np.log(probs) +
                    (1 - actuals) * np.log(1 - probs)
                )
				metrics["CrossEntropy"] = cross_entropy
			all_metrics.append(metrics)
		
		y_true_all = np.array(y_true_all)
		y_pred_all = np.array(y_pred_all)
		global_labels = np.unique(
            np.concatenate([np.unique(y_true_all), np.unique(y_pred_all)])
        )
		global_accuracy = accuracy_score(y_true_all, y_pred_all)
		global_macro_f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
		global_weighted_f1 = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
		
		global_row = {
            "Fold": "Global",
            "Accuracy": global_accuracy,
            "Macro_F1": global_macro_f1,
            "Weighted_F1": global_weighted_f1,
            
        }
		all_metrics.append(global_row)
		eval_df = pd.DataFrame(all_metrics)
		eval_df.to_csv(outputs[eval_name].path, index=False)
		
		precision_g, recall_g, f1_g, support_g = precision_recall_fscore_support(
            y_true_all,
            y_pred_all,
            labels=global_labels,
            zero_division=0
        )
		
		per_class_rows = []
		for i, cls in enumerate(global_labels):
			per_class_rows.append({
                "Class": cls,
                "Precision": precision_g[i],
                "Recall": recall_g[i],
                "F1": f1_g[i],
                "Support": support_g[i],
                "ErrorRate": 1.0 - recall_g[i],
            })
		per_class_df = pd.DataFrame(per_class_rows)
		per_class_df.to_csv(outputs[f"{eval_name}_per_class"].path, index=False)
		cm_global = confusion_matrix(
            y_true_all,
            y_pred_all,
            labels=global_labels
        )
		cm_df = pd.DataFrame(
            cm_global,
            index=global_labels,
            columns=global_labels
        )
		def truncate_cmap(cmap, minval=0.0, maxval=1.0, n=256):
			return mcolors.LinearSegmentedColormap.from_list(
				f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
				cmap(np.linspace(minval, maxval, n))
    )
		cm_df.index.name = "Actual"
		cm_df.columns.name = "Predicted"
		cm_df.to_csv(outputs[f"{eval_name}_confusion_matrix_csv"].path)
		
		fig, ax = plt.subplots(figsize=(6, 5))
		base_cmap = plt.get_cmap("Greens")
		light_cmap = truncate_cmap(base_cmap, 0.15, 0.75)
		im = ax.imshow(
    	cm_global,
    	interpolation="nearest",
    	cmap=light_cmap,
    	vmin=0,
    	vmax=np.max(cm_global) if np.max(cm_global) > 0 else 1)
		ax.set_title("Confusion Matrix")
		ax.set_xlabel("Predicted label")
		ax.set_ylabel("True label")
		ax.set_xticks(np.arange(len(global_labels)))
		ax.set_yticks(np.arange(len(global_labels)))
		ax.set_xticklabels(global_labels, rotation=45, ha="right")
		ax.set_yticklabels(global_labels)
		
		thresh = im.norm(cm_global.max()) * 0.6  
		for i in range(cm_global.shape[0]):
			for j in range(cm_global.shape[1]):
				val = cm_global[i, j]
				ax.text(
            	j, i, val,
            	ha="center", va="center",
            	color="white" if im.norm(val) > thresh else "black"
        	)
		cbar = fig.colorbar(im, ax=ax)
		cbar.set_label("Count")


		fig.tight_layout()
		plt.savefig(
            outputs[f"{eval_name}_confusion_matrix_plot"].path,
            dpi=300,
            bbox_inches="tight"
        )
		plt.close(fig)


	
		
class MulticlassificationPipeline(luigi.WrapperTask):	
	input_dir = luigi.Parameter() 
	output_dir = luigi.Parameter()
	config = luigi.Parameter() 
	tasks = {
		'evaluation': evaluation,
		'lstm' : lstm,
		'randomforest' : randomforest,
		'fullyConnectedNN' : FullyConnectedNN,
		
	}
	task_mapping = {
		'evaluation': ['csv', 'csv'],
		'lstm': ['csv', 'csv'],
		'randomforest'  : ['csv', 'csv'],
		'fullyConnectedNN': ['csv', 'csv'],
		
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
		[MulticlassificationPipeline(
			input_dir=args.input, 
			output_dir=args.output, 
			config=args.config
		)], 
		scheduler_host='localhost', 
		scheduler_port=8082
	)

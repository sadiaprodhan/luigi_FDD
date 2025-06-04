import os 
import argparse
import yaml
import luigi 
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.dates as mdates
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



	
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
class cluster_optics(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		output_files = {}
		file_base = os.path.splitext(os.path.basename(self.input_file[0]))[0]
		output_files[f'cluster_data'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_cluster_data.csv"))
		output_files[f'faults_column'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_faults_column.csv"))
		return output_files
	def run(self):
		
		min_points = self.params['min_points']
		df = pd.read_csv(self.input()[0].path, index_col="Datetime", parse_dates=True)
		#df = df.bfill()  
		label_col = self.params['label_col']
		label_column = df.pop(f"{label_col}") 
		subLabel_col = self.params['sublabel_col']
		sublabel_column = df.pop(f"{subLabel_col}")
		
		scaler = StandardScaler()
		data_= scaler.fit_transform(df)
		df = pd.DataFrame(data_, index=df.index, columns=df.columns)	
		pca = PCA(n_components=3)
		data_reduced = pca.fit_transform(df)
		df =  pd.DataFrame(data_reduced, index=df.index)
	
		eps_value = self.find_k_distance(df,min_points)
		print(f"value  {eps_value}")
		clustering = OPTICS(min_samples=min_points, max_eps=eps_value).fit(df)
		labels = clustering.labels_
		unique_clusters = np.unique(labels[labels != -1])
		number_of_clusters = len(unique_clusters)
		unique_clusters = np.unique(labels[labels != -1])
		number_of_clusters = len(unique_clusters)
		print("Number of clusters (excluding noise):", number_of_clusters)
		reachability = clustering.reachability_[clustering.ordering_]
		thresold = self.params['eps_thresold']

		plt.figure(figsize=(12, 6))
		plt.bar(np.arange(len(reachability)), reachability, color='r', width=1.0)  
		plt.axhline(y=thresold, color='b', linestyle='--', label=f"Threshold = {thresold}")

		plt.title("Reachability Distance Plot (OPTICS)")
		plt.xlabel("Data Points (Ordered)")
		plt.ylabel("Reachability Distance")
		plt.legend(loc='upper right')

		plt.show()
		#labels = cluster_optics_dbscan(reachability=clustering.reachability_,
                              # core_distances=clustering.core_distances_,
                               #ordering=clustering.ordering_,
                               #eps=thresold)
		#unique_clusters = np.unique(labels[labels != -1])
		#number_of_clusters = len(unique_clusters)
		#print("Number of NEW clusters (excluding noise):", number_of_clusters)
		df['Cluster'] = labels
		df['Label']= label_column 
		df['SubLabel']= sublabel_column
		output_files = self.output()
		df.to_csv(output_files[f'cluster_data'].path, index = True)
		label_column.to_csv(output_files[f'faults_column'].path, index = True)


		return None
	def find_k_distance(self,data, k):
		neigh = NearestNeighbors(n_neighbors=k)
		neigh.fit(data)
		distances, indices = neigh.kneighbors(data)
		k_distances = distances[:, k-1]
		k_distances_sorted = np.sort(k_distances)
		knee_locator = KneeLocator(range(len(k_distances_sorted)), k_distances_sorted, curve='convex', direction='increasing')
		epsilon_value = k_distances_sorted[knee_locator.knee]
		plt.figure(figsize=(10, 5))
		plt.plot(k_distances_sorted, marker='o', linestyle='-')
		plt.title(f"{k}-Distance Graph")
		plt.xlabel("Points sorted by distance")
		plt.ylabel(f"Distance to {k}-th nearest neighbor")
		plt.axvline(x=knee_locator.knee, color='r', linestyle='--', label=f'Epsilon = {epsilon_value:.2f}')
		plt.legend()
		plt.grid(True)
		plt.show()
		return epsilon_value
	
class analysis(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		output_files = {}
		file_base = os.path.splitext(os.path.basename(self.input_file[0]))[0]
		output_files[f'sublabel_analysis'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_sublabel_analysis.csv"))
		output_files[f'analysis'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_analysis.csv"))
		output_files[f'noisy_cluster'] = luigi.LocalTarget(os.path.join(self.output_file[0], f"{file_base}_noisy_cluster.csv"))
		return output_files
	def run(self):
		dataframes = {'cluster_data': None, 'faults_column': None}
		for input_target in self.input():
				file_base = os.path.splitext(os.path.basename(input_target.path))[0]
				for key in dataframes.keys():
					if f"{key}" in file_base:
						dataframes[key] = pd.read_csv(input_target.path, index_col="Datetime", parse_dates=True)
		faults= dataframes['cluster_data']['Label'].unique()
		print("faults",faults)
		results = []
		cluster_metrics = []
		sublabel_results = []
		for cl in sorted(dataframes['cluster_data']["Cluster"].unique()): 
			unique_faults =[]
			cluster_data = dataframes['cluster_data'][dataframes['cluster_data']["Cluster"] == cl] 
			fault_counts = cluster_data["Label"].value_counts()
			fault_counts = fault_counts[fault_counts > 0] 
			if len(fault_counts) > 1:
				mean_faults = fault_counts.mean() 
				std_faults = fault_counts.std()  
				unique_faults = len(fault_counts)  
				top_fault = fault_counts.idxmax()  
				top_fault_count = fault_counts.max()  
				cluster_metrics.append({
                "Cluster": f"Cluster {cl}",
                "Summary": f"""Mean Fault Count: {mean_faults}
				Std Dev of Faults: {std_faults}
				Number of Unique Faults: {unique_faults}
				Top Fault Type: {top_fault}
				Top Fault Count: {top_fault_count}"""

              
            })
			
			cluster_total = len(cluster_data)
			fault_percentage = (fault_counts / cluster_total) * 100
			total_fault_counts = dataframes['cluster_data']["Label"].value_counts()
			fault_coverage = (fault_counts / total_fault_counts) * 100
			for fault in faults:
				results.append({
                "Fault": fault,
                "Metric": "Count",
                "Cluster": f"Cluster {cl}",
                "Value": fault_counts.get(fault, "")
            })
				results.append({
                "Fault": fault,
                "Metric": "Percentage in Cluster",
                "Cluster": f"Cluster {cl}",
                "Value": fault_percentage.get(fault, "")
            })
				results.append({
                "Fault": fault,
                "Metric": "Coverage in Dataset",
                "Cluster": f"Cluster {cl}",
                "Value": fault_coverage.get(fault, "")
            })
			sublabel_counts = cluster_data.groupby('SubLabel').size()
			for SubLabel, count in sublabel_counts.items():
				sublabel_results.append({
					"Cluster": f"Cluster {cl}",
					"SubLabel With Count": f"{SubLabel} => {count}"
				})
			
				
		sublabel_df = pd.DataFrame(sublabel_results)
		final_results = pd.DataFrame(results)
		final_results = final_results.pivot(index=["Fault", "Metric"], columns="Cluster", values="Value")
		final_results = final_results.reindex(faults, level="Fault")
		sorted_clusters = sorted(final_results.columns, key=lambda x: int(x.split()[-1]))  # Extract number from "Cluster X"
		final_results = final_results[sorted_clusters]
		output_files = self.output()
		sublabel_df.to_csv(output_files[f'sublabel_analysis'].path)
		final_results.to_csv(output_files[f'analysis'].path)
		cluster_metrics_df = pd.DataFrame(cluster_metrics)
		cluster_metrics_df.to_csv(output_files[f'noisy_cluster'].path)

	
	
class ClusteringPipeline(luigi.WrapperTask):	
	input_dir = luigi.Parameter() 
	output_dir = luigi.Parameter()
	config = luigi.Parameter() 
	tasks = {
		'cluster_optics' : cluster_optics,
		'analysis': analysis

	}
	task_mapping = {
		'cluster_optics': ['csv', 'csv'],
		'analysis': ['csv', 'csv']
		
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
		[ClusteringPipeline(
			input_dir=args.input, 
			output_dir=args.output, 
			config=args.config
		)], 
		scheduler_host='localhost', 
		scheduler_port=8082
	)

import os 
import argparse
import yaml
import luigi 
import pandas as pd
import numpy as np
import joblib
import json
import shutil
from urllib.parse import urlparse, unquote
from sklearn.model_selection import StratifiedShuffleSplit
import shap
import re
import networkx as nx
from rdflib import Graph, Namespace
from rdflib.namespace import RDFS
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from scipy.stats import pointbiserialr
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_docs as tfdocs
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

import tensorflow_docs.modeling


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
'''
def fit_beta_from_samples(x, eps=1e-6):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = np.clip(x, eps, 1 - eps)
    if len(x) < 2:
        
        return 2.0, 8.0
    m = np.mean(x)
    v = np.var(x, ddof=1) + eps
    
    common = m*(1-m)/v - 1
    alpha = max(eps, m*common)
    beta  = max(eps, (1-m)*common)
    return alpha, beta

def log_beta_pdf(x, a, b, eps=1e-12):
    x = np.clip(float(x), eps, 1 - eps)
    from math import lgamma, log
    return (a-1)*log(x) + (b-1)*log(1-x) - (lgamma(a)+lgamma(b)-lgamma(a+b))


def _collect_fold_inputs(targets, fold, keys):
    dfs = {k: None for k in keys}
    for t in targets:
        base = os.path.splitext(os.path.basename(t.path))[0]
        if f"_fold_{fold}" in base:
            for k in keys:
                if k in base:
                    dfs[k] = pd.read_csv(t.path, index_col="Datetime", parse_dates=True)
    if any(v is None for v in dfs.values()):
        missing = [k for k,v in dfs.items() if v is None]
        raise ValueError(f"Missing for fold {fold}: {missing}")
    return dfs
'''




			

class ExplainSHAP(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	
	def output(self):
		outdir = self.output_file[0]
		outs = {}
		for fold in range(1, 6):
			outs[f"shap_raw_fold_{fold}"] = luigi.LocalTarget(
                os.path.join(outdir, f"shap_raw_fold_{fold}.csv")
            )
            
			outs[f"shap_grouped_fold_{fold}"] = luigi.LocalTarget(
                os.path.join(outdir, f"shap_grouped_fold_{fold}.csv")
            )
            
			outs[f"sampled_rows_fold_{fold}"] = luigi.LocalTarget(
                os.path.join(outdir, f"sampled_rows_fold_{fold}.csv")
            )
			outs[f"bg_{fold}"] = luigi.LocalTarget(
                os.path.join(outdir, f"bg_{fold}.csv")
            )
		outs["brick_tree_json"] = luigi.LocalTarget(os.path.join(outdir, "brick_tree_json.json"))
		outs["sig2comp_json"]   = luigi.LocalTarget(os.path.join(outdir, "sig2comp_json.json"))
		return outs
	def _copy_optional_json(self):
		in_paths = [t.path for t in self.input()]
		for name in ("brick_tree_json.json", "sig2comp_json.json"):
			srcs = [p for p in in_paths if os.path.basename(p) == name]
			if srcs:
				key = os.path.splitext(name)[0]
				self.output()[key].makedirs()
				shutil.copy2(srcs[0], self.output()[key].path)
	def _find_inputs_for_fold(self, fold: int):
		model_path = None
		pred_df = None
		Xte = None
		model_stem = self.params.get("model_stem", "model")        
		pred_stem  = self.params.get("pred_stem", "predicted")     
		xte_stem   = self.params.get("xte_stem", "X_test")
		yte_stem   = self.params.get("yte_stem", "Y_test")    

		for t in self.input():
			base = os.path.splitext(os.path.basename(t.path))[0]
			if base == f"{model_stem}_fold_{fold}":
				model_path = t.path
			elif base == f"{pred_stem}_fold_{fold}":
				pred_df = pd.read_csv(t.path, index_col="Datetime", parse_dates=True)
			elif base == f"{xte_stem}_fold_{fold}":
				Xte = pd.read_csv(t.path, index_col="Datetime", parse_dates=True)
			elif base == f"{yte_stem}_fold_{fold}":
				Yte = pd.read_csv(t.path, index_col="Datetime", parse_dates=True)	
		if model_path is None or pred_df is None or Xte is None:
			raise ValueError(f"ExplainSHAPMulticlassJoblib: missing inputs for fold {fold}")
		return model_path, pred_df, Xte, Yte
	def _predict_proba(self, model, X_df: pd.DataFrame) -> np.ndarray:
		if hasattr(model, "predict_proba"):
			proba = np.asarray(model.predict_proba(X_df))
			if proba.ndim != 2:
				raise ValueError(f"predict_proba returned shape {proba.shape}, expected (n,C).")
			return proba
		if hasattr(model, "predict"):
			y = np.asarray(model.predict(X_df))
			if y.ndim == 2 and y.shape[1] >= 2:
				return y
			raise ValueError(
                f"Model.predict returned shape {y.shape}. For multiclass, expected (n,C) probabilities."
            )
		raise ValueError("Loaded model has neither predict_proba nor predict.")
	
	def _get_class_labels(self, model, pred_df: pd.DataFrame) -> list:
		if hasattr(model, "classes_"):
			return list(model.classes_)
		prob_cols = [c for c in pred_df.columns if c.startswith("Prob_")]
		if not prob_cols:
			raise ValueError("Cannot infer class labels: model has no classes_ and pred_df has no Prob_* columns.")
		return [c.replace("Prob_", "", 1) for c in prob_cols]
	
	def _extract_vals_for_class(self, shap_out, class_ix: int, n: int, f: int) -> np.ndarray:
		if isinstance(shap_out, list):
			arr = np.asarray(shap_out[class_ix if class_ix < len(shap_out) else 0])
			if arr.shape != (n, f):
				raise ValueError(f"List SHAP for class {class_ix} has shape {arr.shape}, expected {(n,f)}")
			return arr
		arr = np.asarray(getattr(shap_out, "values", shap_out))
		
		if arr.ndim == 2:
			if arr.shape != (n, f):
				raise ValueError(f"2D SHAP has shape {arr.shape}, expected {(n,f)}")
			return arr
		
		if arr.ndim == 3:
            
			if arr.shape[0] == n and arr.shape[1] == f:
				return arr[:, :, class_ix if arr.shape[2] > class_ix else 0]
			if arr.shape[0] == n and arr.shape[2] == f:
				return arr[:, class_ix if arr.shape[1] > class_ix else 0, :]
			if arr.shape[1] == n and arr.shape[2] == f:
				return arr[class_ix if arr.shape[0] > class_ix else 0, :, :]
		raise ValueError(f"Unexpected SHAP shape {arr.shape}; cannot extract class {class_ix}.")
	def _base_signal_name(self, col: str) -> str:
		if re.match(r"^year_\d+$", col):
			return "year"
		if re.match(r"^month_\d{1,2}$", col):
			return "month"
		if re.match(r"^day_\d{1,2}$", col):
			return "day"
		if re.match(r"^hour_\d{1,2}$", col):
			return "hour"
		col2 = re.sub(r"_lag_\d+$", "", col)
		col2 = re.sub(r"_(roll|rolling)_(mean|std|min|max|median)_\d+$", "", col2)
		col2 = re.sub(r"_roll(mean|std|min|max|median)_\d+$", "", col2)
		if col2 == col:
			parts = col.split("_")
			if len(parts) >= 2 and len(parts[-1]) <= 10:
				if parts[0] in {"dow", "weekday", "season", "mode"}:
					return "_".join(parts[:-1])
		return col
	def _group_shap_by_signal(self, shap_df: pd.DataFrame) -> pd.DataFrame:
		feat_cols = [c for c in shap_df.columns if c not in ("Predicted", "Prob_PredClass")]
		groups = {}
		for c in feat_cols:
			base = self._base_signal_name(c)
			groups.setdefault(base, []).append(c)
		grouped = pd.DataFrame(index=shap_df.index)
		for base, cols in groups.items():
			grouped[base] = shap_df[cols].sum(axis=1)
		grouped["Predicted"] = shap_df["Predicted"].values
		grouped["Prob_PredClass"] = shap_df["Prob_PredClass"].values
		return grouped

    
	def _predclass_prob_series(self, pred_df: pd.DataFrame, pred_labels: pd.Series) -> pd.Series:
		probs = pd.Series(index=pred_df.index, data=np.nan, name="Prob_PredClass")
        
		for idx, lab in pred_labels.astype(str).items():
			col = f"Prob_{lab}"
			if col in pred_df.columns:
				probs.loc[idx] = pred_df.loc[idx, col]
		return probs
	def _sample_rows_per_class(self,pred_df: pd.DataFrame,Xte: pd.DataFrame, normal_label: str, conf_thresh: float, per_class_n: int = 200,
	method: str = "top_conf",   random_state: int = 0,) -> pd.DatetimeIndex:
		if "Predicted" not in pred_df.columns:
			raise ValueError("pred_df must contain 'Predicted'.")
		df = pred_df.loc[pred_df.index.intersection(Xte.index)].copy()
		df["Predicted"] = df["Predicted"].astype(str)
		df = df[df["Predicted"] != str(normal_label)]
		if df.empty:
			return pd.DatetimeIndex([])
		rng = np.random.RandomState(random_state)
		sampled = []
		for lab, g in df.groupby("Predicted"):
			n = len(g)
			if n <= per_class_n:
				chosen = g.index
			else:
				g2 = g
				g_thr = None 
				if "Prob_PredClass" in g2.columns and g2["Prob_PredClass"].notna().any():
					g_thr = g2[g2["Prob_PredClass"].astype(float) >= conf_thresh]
				if g_thr is not None and len(g_thr) >= max(5, per_class_n // 4):
					g2 = g_thr
				if method == "top_conf" and "Prob_PredClass" in g2.columns and g2["Prob_PredClass"].notna().any():
					chosen = g2.sort_values("Prob_PredClass", ascending=False).head(per_class_n).index
				else:
					chosen = g2.sample(n=per_class_n, random_state=random_state).index

			sampled.extend(list(chosen))
		seen = set()
		sampled_unique = [x for x in sampled if not (x in seen or seen.add(x))]
		return pd.DatetimeIndex(sampled_unique)
	
	
	def run(self):
		self._copy_optional_json()
		normal_label = self.params.get("normal_label", "Normal")  
		conf_thresh  = float(self.params.get("confidence_threshold", 0.80))  
		bg_size      = int(self.params.get("bg_size", 200))
		explainer_kind = self.params.get("explainer_kind", "auto").lower()  
		
		
		for fold in range(1, 6):
			model_path, pred_df, Xte, Yte = self._find_inputs_for_fold(fold)
			model = joblib.load(model_path)
			if "Predicted" not in pred_df.columns:
				raise ValueError("pred_df must contain a 'Predicted' column.")
			
			pred_df = pred_df.sort_index()
			Xte = Xte.sort_index()
			Yte = Yte.sort_index()
			
			pred_labels = pred_df["Predicted"].astype(str)
			fault_mask = pred_labels != str(normal_label)
			
			prob_pred = self._predclass_prob_series(pred_df, pred_labels)
			pred_df["Prob_PredClass"] = prob_pred
			
			
			fault_df = pred_df.loc[fault_mask].copy()
			fault_df = fault_df.loc[fault_df.index.intersection(Xte.index)]

			per_class_n = int(self.params.get("per_class_n", 200))  
			sample_method = self.params.get("sample_method", "top_conf").lower()  
			sampled_rows = self._sample_rows_per_class(
				pred_df=pred_df,Xte=fault_df,
				normal_label=normal_label,
				conf_thresh = conf_thresh,
				per_class_n=per_class_n,
				method=sample_method,
				random_state=0)
			Xsub = Xte.loc[sampled_rows]
           
			out_rows = self.output()[f"sampled_rows_fold_{fold}"]
			out_rows.makedirs()
			pd.DataFrame({
                "Datetime": sampled_rows,
                "Predicted": pred_df.reindex(sampled_rows)["Predicted"].astype(str).values,
                "Prob_PredClass": pred_df.reindex(sampled_rows)["Prob_PredClass"].values
            }).to_csv(out_rows.path, index=False)

            
			if len(Xsub) == 0:
				self.output()[f"shap_raw_fold_{fold}"].makedirs()
				self.output()[f"shap_grouped_fold_{fold}"].makedirs()
				pd.DataFrame(index=[], columns=list(Xte.columns) + ["Predicted", "Prob_PredClass"]).to_csv(
                    self.output()[f"shap_raw_fold_{fold}"].path, index=True
                )
				pd.DataFrame(index=[], columns=["Predicted", "Prob_PredClass"]).to_csv(
                    self.output()[f"shap_grouped_fold_{fold}"].path, index=True
                )
				continue

            
			#bg = Xte.sample(n=min(max_bg, len(Xte)), random_state=0)
			bg, _, y_bg, _ = train_test_split(Xte,Yte,
									   train_size=min(bg_size, len(Xte)),
									   stratify=Yte,random_state=42
)
			
			class_labels = self._get_class_labels(model, pred_df)
			label_to_ix = {str(lbl): i for i, lbl in enumerate(class_labels)}

            
			def f_predict(x_np: np.ndarray) -> np.ndarray:
				x_df = pd.DataFrame(x_np, columns=Xte.columns)
				return self._predict_proba(model, x_df)

            
			if explainer_kind == "kernel":
				explainer = shap.KernelExplainer(f_predict, bg.values)
				shap_out = explainer.shap_values(Xsub.values)
			else:
				try:
					explainer = shap.Explainer(f_predict, bg.values, feature_names=list(Xte.columns))
					shap_out = explainer(Xsub.values)
				except Exception:
					explainer = shap.KernelExplainer(f_predict, bg.values)
					shap_out = explainer.shap_values(Xsub.values)
			n, f = Xsub.shape
			vals_faultsignal = np.zeros((n, f), dtype=float)
			
			pred_lab_sub = pred_df.reindex(Xsub.index)["Predicted"].astype(str).values
			prob_sub = pred_df.reindex(Xsub.index)["Prob_PredClass"].values
			
			for i, lab in enumerate(pred_lab_sub):
				if lab not in label_to_ix:
					raise ValueError(f"Predicted label '{lab}' not in model classes {class_labels}")
				cix = label_to_ix[lab]
				class_vals = self._extract_vals_for_class(shap_out, cix, n, f)  
				vals_faultsignal[i, :] = class_vals[i, :]

            
			shap_raw = pd.DataFrame(vals_faultsignal, index=Xsub.index, columns=Xsub.columns)
			shap_raw["Predicted"] = pred_lab_sub
			shap_raw["Prob_PredClass"] = prob_sub
			print(bg.index[:5])
			print(y_bg.index[:5])
			print(bg.index.equals(y_bg.index))
			shap_grouped = self._group_shap_by_signal(shap_raw)
			bg_df = bg.copy()
			bg_df["Label"] =  y_bg.iloc[:, 0].values


            
			self.output()[f"shap_raw_fold_{fold}"].makedirs()
		
			self.output()[f"shap_grouped_fold_{fold}"].makedirs()
			self.output()[f"bg_{fold}"].makedirs()
			
			shap_raw.to_csv(self.output()[f"shap_raw_fold_{fold}"].path, index=True, index_label="Datetime")
			shap_grouped.to_csv(self.output()[f"shap_grouped_fold_{fold}"].path, index=True, index_label="Datetime")
			bg_df.to_csv(self.output()[f"bg_{fold}"].path, index=True, index_label="Datetime")


class TopKSensorsByFault(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	
	def output(self):
		outdir = self.output_file[0]
		return {
            "with_time": luigi.LocalTarget(os.path.join(outdir, "topk_sensors_by_fault_with_time.csv")),
        	"no_time":   luigi.LocalTarget(os.path.join(outdir, "topk_sensors_by_fault_no_time.csv")),
        	"mean_abs_shap_all_features": luigi.LocalTarget(os.path.join(outdir, "mean_abs_shap_all_features.csv")),
			"brick_tree_json": luigi.LocalTarget(os.path.join(outdir, "brick_tree_json.json")),
        	"sig2comp_json":   luigi.LocalTarget(os.path.join(outdir, "sig2comp_json.json"))
		}
	def _copy_optional_json(self):
		in_paths = [t.path for t in self.input()]
		for name in ("brick_tree_json.json", "sig2comp_json.json"):
			srcs = [p for p in in_paths if os.path.basename(p) == name]
			if srcs:
				key = os.path.splitext(name)[0]
				self.output()[key].makedirs()
				shutil.copy2(srcs[0], self.output()[key].path)
	def _load_all_grouped_shap(self) -> pd.DataFrame:
		shap_paths = []
		for t in self.input():
			base = os.path.splitext(os.path.basename(t.path))[0]
			if base.startswith("shap_grouped_fold_"):
				shap_paths.append(t.path)
		if not shap_paths:
			raise ValueError("TopKSensorsByFault: no shap_grouped_fold_*.csv found in inputs")
		
		dfs = []
		for p in sorted(shap_paths):
			df = pd.read_csv(p, index_col="Datetime", parse_dates=True)
			if "Predicted" not in df.columns:
				raise ValueError(f"TopKSensorsByFault: missing Predicted in {p}")
			dfs.append(df)
		return pd.concat(dfs, axis=0)
	
	def _compute_topk(self, all_df: pd.DataFrame, drop_time: bool) -> pd.DataFrame:
		normal_label = str(self.params.get("normal_label", "Unfaulted"))
		top_k = int(self.params.get("top_k", 20))
		meta_cols = {"Predicted", "Prob_PredClass"}
		feat_cols = [c for c in all_df.columns if c not in meta_cols]

        
		if drop_time:
			feat_cols = [c for c in feat_cols if c not in ("year", "month", "day", "hour")]
			
		rows = []
		for fault, g in all_df.groupby(all_df["Predicted"].astype(str)):
			if str(fault) == normal_label:
				continue
			mean_abs = g[feat_cols].abs().mean(axis=0).sort_values(ascending=False)
			top = mean_abs.head(top_k)
			denom = float(top.sum())
			if denom == 0.0:
				pct = (top * 0.0)  
			else:
				pct = (top / denom) * 100.0
			
			for feat in top.index:
				rows.append({
                    "fault_class": str(fault),
                    "feature": feat,
                    "mean_abs_shap": float(top.loc[feat]),
                    "pct_within_topk": float(pct.loc[feat]),
                    "n_samples_fault": int(len(g))
                })
		out_df = pd.DataFrame(rows)
		if not out_df.empty:
			out_df["rank"] = (
                out_df.sort_values(["fault_class", "mean_abs_shap"], ascending=[True, False])
                      .groupby("fault_class")
                      .cumcount() + 1
            )
		return out_df
	def _mean_abs_all_features_with_top5(self, all_df: pd.DataFrame) -> pd.DataFrame:
		normal_label = str(self.params.get("normal_label", "Unfaulted"))
		meta_cols = {"Predicted", "Prob_PredClass"}
		feat_cols = [c for c in all_df.columns if c not in meta_cols]  
		rows = []
		for fault, g in all_df.groupby(all_df["Predicted"].astype(str)):
			if str(fault) == normal_label:
				continue
			mean_abs = g[feat_cols].abs().mean(axis=0)  
			top5 = mean_abs.sort_values(ascending=False).head(5)
			row = {"fault": str(fault), "top5_features": "; ".join([f"{k} ({v:.6f})" for k, v in top5.items()])}
			row.update(mean_abs.to_dict())
			rows.append(row)
		if not rows:
			return pd.DataFrame(columns=["fault", "top5_features"] + sorted(feat_cols))
		df = pd.DataFrame(rows)
		features_sorted = sorted([c for c in df.columns if c not in ("fault", "top5_features")])
		df = df[["fault", "top5_features"] + features_sorted]
		return df
	
	def run(self):
		self._copy_optional_json()
		all_df = self._load_all_grouped_shap()
		out_with = self._compute_topk(all_df, drop_time=False)
		out_no   = self._compute_topk(all_df, drop_time=True)
		mean_all = self._mean_abs_all_features_with_top5(all_df)
		outs = self.output()
		outs["mean_abs_shap_all_features"].makedirs()
		outs["with_time"].makedirs()
		outs["no_time"].makedirs()
		out_with.to_csv(outs["with_time"].path, index=False)
		mean_all.to_csv(outs["mean_abs_shap_all_features"].path, index=False)
		out_no.to_csv(outs["no_time"].path, index=False)

class TopComponentsByFaultFromTopSensors(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		outdir = self.output_file[0]
		return {
            "with_time": luigi.LocalTarget(os.path.join(outdir, "topk_components_by_fault_with_time.csv")),
            "no_time":   luigi.LocalTarget(os.path.join(outdir, "topk_components_by_fault_no_time.csv")),
			"report_with_time": luigi.LocalTarget(os.path.join(outdir, "report_with_time.csv")),
            "report_no_time":   luigi.LocalTarget(os.path.join(outdir, "report_no_time.csv")),

        }
	
	def _load_sig2comp(self):
		path = None
		for t in self.input():
			if os.path.basename(t.path) == "sig2comp_json.json":
				path = t.path
				break
		if path is None:
			raise ValueError("TopKComponentsByFaultFromTopSensors: missing sig2comp_json.json in inputs")
		
		with open(path, "r") as f:
			mapping = json.load(f)
			
		norm = {}
		for sig, comp in mapping.items():
			if comp is None:
				continue
			if isinstance(comp, list):
				if len(comp) == 1:
					norm[str(sig)] = str(comp[0])
				else:
					raise ValueError(
                    f"Signal '{sig}' maps to multiple components {comp} but 1->1 mapping assumed."
                )
			else:
				norm[str(sig)] = str(comp)
		return norm
	def _load_topk_sensors(self, which: str) -> pd.DataFrame:
		target_name = f"topk_sensors_by_fault_{which}.csv"
		def iter_targets(inp):
			if isinstance(inp, dict):
				for v in inp.values():
					yield from iter_targets(v)
			else:
				for t in inp:
					yield t
		seen = []
		for t in iter_targets(self.input()):
			seen.append(os.path.basename(t.path))
			if os.path.basename(t.path) == target_name:
				return pd.read_csv(t.path)
		raise ValueError(
        f"TopKComponentsByFaultFromTopSensors: missing {target_name} in inputs. "
        f"Inputs seen: {seen}"
    )
	def _compute_components(self, topk_df: pd.DataFrame, sig2comp: dict) -> pd.DataFrame:
		rows = []
		for fault, g in topk_df.groupby("fault_class"):
			comp_pct = {}   
			comp_mean = {}  
			for _, r in g.iterrows():
				sig = str(r["feature"])
				if sig not in sig2comp:
					continue
				comp = sig2comp[sig]
				pct = float(r["pct_within_topk"])      
				mean_abs = float(r["mean_abs_shap"])
				comp_pct[comp] = comp_pct.get(comp, 0.0) + pct
				comp_mean[comp] = comp_mean.get(comp, 0.0) + mean_abs
			if not comp_pct:
				continue
			comp_df = pd.DataFrame({
                "component": list(comp_pct.keys()),
                "pct_from_topk_sensors": list(comp_pct.values()),
                "mean_abs_shap_from_topk_sensors": [comp_mean[c] for c in comp_pct.keys()],
            })
			comp_df = comp_df.sort_values("pct_from_topk_sensors", ascending=False).reset_index(drop=True)
			comp_df["rank"] = np.arange(1, len(comp_df) + 1)
			comp_df.insert(0, "fault_class", fault)

            
			denom = float(comp_df["pct_from_topk_sensors"].sum())
			comp_df["pct_within_all_mapped_components"] = 0.0 if denom == 0.0 else (
                comp_df["pct_from_topk_sensors"] / denom * 100.0
            )
			rows.append(comp_df)
		if rows:
			return pd.concat(rows, axis=0, ignore_index=True)
		return pd.DataFrame(columns=[
            "fault_class", "component",
            "pct_from_topk_sensors", "mean_abs_shap_from_topk_sensors",
            "pct_within_all_mapped_components", "rank"
        ])
	def _build_report(self, topk_df: pd.DataFrame, sig2comp: dict, comp_df: pd.DataFrame) -> pd.DataFrame:
		if topk_df.empty or comp_df.empty:
			return pd.DataFrame(columns=[
                "fault_class", "component", "component_pct",
                "sensors_contributing"
            ])
		rows = []
        
		topk_df = topk_df.copy()
		topk_df["fault_class"] = topk_df["fault_class"].astype(str)
		topk_df["feature"] = topk_df["feature"].astype(str)
		for fault, g in topk_df.groupby("fault_class"):
			comp_to_sensors = {}  
			for _, r in g.iterrows():
				sensor = str(r["feature"])
				if sensor not in sig2comp:
					continue
				comp = sig2comp[sensor]
				pct = float(r["pct_within_topk"])
				comp_to_sensors.setdefault(comp, []).append((sensor, pct))
			sub_comp = comp_df[comp_df["fault_class"].astype(str) == str(fault)].copy()
			sub_comp = sub_comp.sort_values("pct_from_topk_sensors", ascending=False)
			for _, crow in sub_comp.iterrows():
				comp = str(crow["component"])
				comp_pct = float(crow["pct_within_all_mapped_components"])  
				sensors = comp_to_sensors.get(comp, [])

                
				sensors = sorted(sensors, key=lambda x: x[1], reverse=True)
				sensors_str = "; ".join([f"{s} ({p:.2f}%)" for s, p in sensors]) if sensors else ""
				rows.append({
                    "fault_class": str(fault),
                    "component": comp,
                    "component_pct": round(comp_pct, 4),
                    "sensors_contributing": sensors_str
                })
		return pd.DataFrame(rows)
	def run(self):
		sig2comp = self._load_sig2comp()
		top_with = self._load_topk_sensors("with_time")
		top_no   = self._load_topk_sensors("no_time")
		comp_with = self._compute_components(top_with, sig2comp)
		comp_no   = self._compute_components(top_no, sig2comp)
		report_with = self._build_report(top_with, sig2comp, comp_with)
		report_no   = self._build_report(top_no, sig2comp, comp_no)
		outs = self.output()
		outs["with_time"].makedirs()
		outs["no_time"].makedirs()
		comp_with.to_csv(outs["with_time"].path, index=False)
		comp_no.to_csv(outs["no_time"].path, index=False)
		report_with.to_csv(outs["report_with_time"].path, index=False)
		report_no.to_csv(outs["report_no_time"].path, index=False)

class BuildBrickTree(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]
	def output(self):
		outdir = self.output_file[0]
		outs = {
            "brick_tree_json": luigi.LocalTarget(os.path.join(outdir, "brick_tree_json.json")),
            "sig2comp_json":  luigi.LocalTarget(os.path.join(outdir, "sig2comp_json.json")),
        }
        
		for t in self.input_file:
			base = os.path.basename(t)
			outs[f"passthrough_{base}"] = luigi.LocalTarget(os.path.join(outdir, base))
		return outs

	def run(self):
		def local_id(u: str) -> str:
			s = str(u)
			if "#" in s:
				tail = s.rsplit("#", 1)[1]
				return unquote(tail) if tail else s
			p = urlparse(s)
			tail = os.path.basename(p.path)
			return unquote(tail) if tail else s
		outdir = self.output_file[0]
		os.makedirs(outdir, exist_ok=True)
		ttl_path = None
		for t in self.input():
			if t.path.lower().endswith((".ttl", ".turtle")):
				ttl_path = t.path
				break
		if ttl_path is None:
			raise ValueError("BuildBrickTree: TTL file not found in input bundle")
		BRICK = Namespace("https://brickschema.org/schema/Brick#")
		g = Graph()
		g.parse(ttl_path, format="turtle")
		include_points = True

        
		q_equips = """
        SELECT DISTINCT ?e ?t ?label WHERE {
		
          ?e a ?t .
          FILTER(CONTAINS(STR(?t), "Brick#")) .
          OPTIONAL { ?e rdfs:label ?label . }
        }"""
		nodes = {}  
		for e, t, lab in g.query(q_equips, initNs={"rdfs": RDFS}):
			e_uri = str(e)
			e_id = local_id(e_uri)  # short id only
			tname = str(t).split("#")[-1]
			nodes[e_id] = {
                "id": e_id,                 # SHORT id
                "uri": e_uri,               # full URI kept for traceability
                "kind": tname,
                "brick_type": tname,
                "label": str(lab).strip() if lab else e_id,
            }
		edges = []


		q_parts = f"SELECT ?p ?c WHERE {{ ?p <{BRICK.hasPart}> ?c . }}"
		for p, c in g.query(q_parts):
			p_id = local_id(str(p))
			c_id = local_id(str(c))
			if p_id in nodes and c_id in nodes:
				edges.append({"src": p_id, "dst": c_id, "rel": "hasPart"})

        
		sig2comp = {}  
		q_points = f"""
        SELECT ?eq ?pt ?ptype ?lab WHERE {{
          {{ ?eq <{BRICK.hasPoint}> ?pt . }}
          UNION
          {{ ?pt <{BRICK.isPointOf}> ?eq . }}
          ?pt a ?ptype .
          OPTIONAL {{ ?pt rdfs:label ?lab . }}
        }}"""
		for eq, pt, ptype, lab in g.query(q_points, initNs={"rdfs": RDFS}):
			eq_uri = str(eq)
			pt_uri = str(pt)
			eq_id = local_id(eq_uri)
			pt_id = local_id(pt_uri)
			ptype_name = str(ptype).split("#")[-1]
			sig_label = (str(lab).strip() if lab else "") or pt_id

			if include_points:
				if pt_id not in nodes:
					nodes[pt_id] = {
                        "id": pt_id,         
                        "uri": pt_uri,       
                        "kind": ptype_name,
                        "brick_type": ptype_name,
                        "label": sig_label,
                    }
				if eq_id in nodes:
					edges.append({"src": eq_id, "dst": pt_id, "rel": "hasPoint"})

            
			if eq_id in nodes and sig_label:
				sig2comp.setdefault(sig_label, []).append(eq_id)
		self.output()["brick_tree_json"].makedirs()
		with self.output()["brick_tree_json"].open("w") as f:
			json.dump({"nodes": list(nodes.values()), "edges": edges}, f, indent=2)
		self.output()["sig2comp_json"].makedirs()
		with self.output()["sig2comp_json"].open("w") as f:
			json.dump(sig2comp, f, indent=2)
        
		for t in self.input():
			dst = os.path.join(outdir, os.path.basename(t.path))
			shutil.copy2(t.path, dst)

class CorrelationAnalysisTopKSensors(OneToOneTask):
	def requires(self):
		return [File(file=f) for f in self.input_file]

	def output(self):
		outdir = self.output_file[0]
		return {
			"correlation_all": luigi.LocalTarget(os.path.join(outdir, "main_correlation_topk_merged_all.csv")),
			"correlation_topk_summary": luigi.LocalTarget(os.path.join(outdir, "main_correlation_topk_merged_summary.csv"))
		}

	def _load_topk_sensors(self) -> pd.DataFrame:
		use_time = bool(self.params.get("use_time_features", False))
		target_name = "topk_sensors_by_fault_with_time.csv" if use_time else "topk_sensors_by_fault_no_time.csv"

		for t in self.input():
			if os.path.basename(t.path) == target_name:
				df = pd.read_csv(t.path)
				required = {"fault_class", "feature", "rank"}
				missing = required - set(df.columns)
				if missing:
					raise ValueError(f"Top-k sensor file missing columns: {missing}")
				return df

		raise ValueError(f"CorrelationAnalysisTopKSensors: missing {target_name}")

	def _load_all_test_folds(self):
		xte_stem = self.params.get("xte_stem", "SDAHU_FULL_M_CLASS_X_test")
		yte_stem = self.params.get("yte_stem", "SDAHU_FULL_M_CLASS_Y_test")

		x_list = []
		y_list = []

		for fold in range(1, 6):
			xdf = None
			ydf = None

			for t in self.input():
				base = os.path.splitext(os.path.basename(t.path))[0]
				if base == f"{xte_stem}_fold_{fold}":
					xdf = pd.read_csv(t.path, index_col="Datetime", parse_dates=True)
				elif base == f"{yte_stem}_fold_{fold}":
					ydf = pd.read_csv(t.path, index_col="Datetime", parse_dates=True)

			if xdf is None or ydf is None:
				raise ValueError(f"Missing X_test or Y_test for fold {fold}")

			x_list.append(xdf.sort_index())
			y_list.append(ydf.sort_index())

		x_all = pd.concat(x_list, axis=0)
		y_all = pd.concat(y_list, axis=0)
		return x_all, y_all

	def _extract_label_series(self, ydf: pd.DataFrame) -> pd.Series:
		label_col = self.params.get("label_column", None)

		if label_col is not None:
			if label_col not in ydf.columns:
				raise ValueError(f"Label column '{label_col}' not found in Y dataframe columns {list(ydf.columns)}")
			return ydf[label_col]

		if ydf.shape[1] == 1:
			return ydf.iloc[:, 0]

		for candidate in ["Label"]:
			if candidate in ydf.columns:
				return ydf[candidate]

		raise ValueError(
			f"Could not determine label column from Y dataframe columns {list(ydf.columns)}. "
			f"Set params['label_column'] explicitly."
		)

	def run(self):
		top_k = int(self.params.get("top_k", 5))
		normal_label = str(self.params.get("normal_label", "Unfaulted"))

		topk_df = self._load_topk_sensors()
		topk_df = topk_df[topk_df["rank"] <= top_k].copy()

		x_all, y_all = self._load_all_test_folds()
		y_series = self._extract_label_series(y_all).astype(str)

		common_idx = x_all.index.intersection(y_series.index)
		x_all = x_all.loc[common_idx].copy()
		y_series = y_series.loc[common_idx].copy()

		rows = []
		print("X columns:", list(x_all.columns))
		print("Y unique labels:", y_series.astype(str).unique())
		print("Topk fault classes:", topk_df["fault_class"].astype(str).unique())
		print("Topk features:", topk_df["feature"].astype(str).unique())
		print("Topk rows:", len(topk_df))

		for fault, g in topk_df.groupby("fault_class"):
			fault = str(fault)
			if fault == normal_label:
				continue

			sensors = [str(s) for s in g["feature"].tolist() if str(s) in x_all.columns]
			if len(sensors) == 0:
				continue

			y_bin = (y_series == fault).astype(int)
			print("FAULT:", fault)
			print("Sensors found:", sensors)
			print("Positive samples:", int(y_bin.sum()))

			

			for sensor in sensors:
				x = pd.to_numeric(x_all[sensor], errors="coerce")
				valid = x.notna() & y_bin.notna()
				x_valid = x.loc[valid]
				y_valid = y_bin.loc[valid]
				print("FAULT:", fault, "| SENSOR:", sensor)
				print("  valid rows:", len(x_valid))
				print("  x unique:", x_valid.nunique())
				print("  y unique:", y_valid.nunique())
				print("  y counts:", y_valid.value_counts().to_dict())
				if x_valid.nunique() <= 1:
					print("  SKIP: x_valid constant")
					continue
				if y_valid.nunique() <= 1:
					print("  SKIP: y_valid single class")
					continue
				try:
					corr, pval = pointbiserialr(y_valid, x_valid)
					print("  corr:", corr, "pval:", pval)
				except Exception as e:
					print("  SKIP: correlation failed:", repr(e))
					continue
				rows.append({
					"fault_class": fault,
					"feature": sensor,
					"correlation": float(corr),
					"abs_correlation": float(abs(corr)),
					"p_value": float(pval),
					"n_samples": int(valid.sum()),
					"n_positive": int(y_valid.sum())
					})
				print("  APPENDED")
		all_df = pd.DataFrame(rows)

		if all_df.empty:
			summary_df = pd.DataFrame(columns=[
				"fault_class", "feature", "correlation", "abs_correlation",
				"p_value", "n_samples", "n_positive", "rank_by_abs_correlation"
			])
		else:
			all_df = all_df.sort_values(["fault_class", "abs_correlation"], ascending=[True, False]).reset_index(drop=True)
			all_df["rank_by_abs_correlation"] = all_df.groupby("fault_class").cumcount() + 1
			summary_df = all_df.copy()
		print("Final rows written:", len(all_df))	
		all_df.to_csv(self.output()["correlation_all"].path, index=False)
		summary_df.to_csv(self.output()["correlation_topk_summary"].path, index=False)
		
class BrickPipeline(luigi.WrapperTask):	
	input_dir = luigi.Parameter() 
	output_dir = luigi.Parameter()
	config = luigi.Parameter() 
	tasks = {
		'BuildBrickTree': BuildBrickTree,
		'ExplainSHAP': ExplainSHAP,
		'TopKSensorsByFault':TopKSensorsByFault,
		'TopComponentsByFaultFromTopSensors': TopComponentsByFaultFromTopSensors,
		'CorrelationAnalysisTopKSensors': CorrelationAnalysisTopKSensors

	}
	task_mapping = {
		'BuildBrickTree':['any','any'],
		'ExplainSHAP': ['any', 'any'],
		'TopKSensorsByFault': ['any', 'any'],
		'TopComponentsByFaultFromTopSensors': ['any', 'any'],
		'CorrelationAnalysisTopKSensors' : ['any', 'any']
		
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
		allow_all = file_format in (None, 'any', '*')
		if isinstance(file_format, (list, tuple, set)):
			exts = {str(x).lower().lstrip('.') for x in file_format}
		elif allow_all:
			exts = None
		else:
			exts = {str(file_format).lower().lstrip('.')}
		for root, _, files in os.walk(directory):
			for file in files:
				if allow_all:
					target_files.append(os.path.join(root, file))
				else:
					ext = os.path.splitext(file)[1].lower().lstrip('.')
					if ext in exts:
						target_files.append(os.path.join(root, file))
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
		[BrickPipeline(
			input_dir=args.input, 
			output_dir=args.output, 
			config=args.config
		)], 
		scheduler_host='localhost', 
		scheduler_port=8082
	)

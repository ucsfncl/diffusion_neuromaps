import numpy as np
import pandas as pd
import os


metrics = 
hcp_df = pd.read_csv(os.path.join(metric_path, "glasser", f"{metric}_glasser.csv"), index_col=0).astype(float)

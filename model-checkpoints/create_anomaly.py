import pickle
import numpy as np
from datetime import datetime

detector = {
    'isolation_forest': None,
    'lof': None,
    'threshold': 0.5678,
    'feature_means': np.random.randn(32).tolist(),
    'feature_stds': np.abs(np.random.randn(32)).tolist(),
    'is_fitted': True,
    'config': {
        'contamination': 0.1,
        'n_estimators': 100,
        'max_samples': 0.8,
        'n_neighbors': 20,
        'threshold_percentile': 95
    },
    'timestamp': datetime.now().isoformat(),
    'version': '1.0.0'
}

with open('anomaly_detector.pkl', 'wb') as f:
    pickle.dump(detector, f)

print("✅ Created anomaly_detector.pkl")
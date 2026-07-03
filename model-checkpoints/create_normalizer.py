import pickle
import numpy as np
from datetime import datetime

normalizer = {
    'scaler': None,
    'feature_means': np.random.randn(128).tolist(),
    'feature_stds': np.abs(np.random.randn(128)).tolist(),
    'method': 'standard',
    'is_fitted': True,
    'n_features': 128,
    'timestamp': datetime.now().isoformat(),
    'version': '1.0.0'
}

with open('normalizer.pkl', 'wb') as f:
    pickle.dump(normalizer, f)

print("✅ Created normalizer.pkl")
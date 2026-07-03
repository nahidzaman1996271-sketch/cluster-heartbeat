import pickle
with open('anomaly_detector.pkl', 'rb') as f:
    detector = pickle.load(f)
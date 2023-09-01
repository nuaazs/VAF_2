import torch
from sklearn.preprocessing import MinMaxScaler

def calibrate_scores(qmf_model, asnorm_scores, qmf_features):
    # Normalize SNR features using Max-Min normalization
    snr_features = qmf_features[:, [4, 5]]  # SNR features are at indices 4 and 5
    scaler = MinMaxScaler()
    snr_features_normalized = scaler.transform(snr_features)
    qmf_features[:, [4, 5]] = torch.from_numpy(snr_features_normalized)

    qmf_model.eval()
    with torch.no_grad():
        qmf_scores = qmf_model(qmf_features).squeeze()
    
    # Apply score calibration
    calibrated_scores = asnorm_scores * qmf_scores

    return calibrated_scores

# Example usage
asnorm_scores = torch.randn(100)  # Replace with your actual AS-Norm scores
qmf_features = torch.randn(100, 7)  # Replace with your actual QMF features

qmf_model = QMFModel(input_size=qmf_features.shape[1])
qmf_model.load_state_dict(torch.load('qmf_model.pth'))

calibrated_scores = calibrate_scores(qmf_model, asnorm_scores, qmf_features)

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

class QMFModel(nn.Module):
    def __init__(self, input_size):
        super(QMFModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_qmf(qmf_features, scores):
    # Normalize SNR features using Max-Min normalization
    snr_features = qmf_features[:, [4, 5]]  # SNR features are at indices 4 and 5
    scaler = MinMaxScaler()
    snr_features_normalized = scaler.fit_transform(snr_features)
    qmf_features[:, [4, 5]] = torch.from_numpy(snr_features_normalized)
    
    # Train QMF model
    qmf_model = QMFModel(input_size=qmf_features.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(qmf_model.parameters(), lr=0.01)
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = qmf_model(qmf_features)
        loss = criterion(outputs, scores)
        loss.backward()
        optimizer.step()

    return qmf_model

# Example usage
qmf_features = torch.randn(100, 7)  # Replace with your actual QMF features
scores = torch.randn(100)  # Replace with your actual scores
qmf_model = train_qmf(qmf_features, scores)
torch.save(qmf_model.state_dict(), 'qmf_model.pth')

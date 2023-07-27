import torch
import sys
sys.path.append("/home/zhaosheng/3D-Speaker")
from speakerlab.models.conformer.encoder import ConformerEncoder
from speakerlab.models.conformer.ECAPA_TDNN import AttentiveStatisticsPooling
from speakerlab.models.conformer.ECAPA_TDNN import BatchNorm1d

class Conformer(torch.nn.Module):
    def __init__(self, feat_dim=80, num_blocks=6, output_size=256, embedding_size=192, input_layer="conv2d", 
            pos_enc_layer_type="rel_pos"):
        super(Conformer, self).__init__()
        self.conformer = ConformerEncoder(input_size=feat_dim, num_blocks=num_blocks, 
                output_size=output_size, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)
        self.pooling = AttentiveStatisticsPooling(output_size)
        self.bn = BatchNorm1d(input_size=output_size*2)
        self.fc = torch.nn.Linear(output_size*2, embedding_size)
    
    def forward(self, feat):
        # feat = feat.squeeze(1).permute(0, 2, 1)
        # feat = feat.permute(0, 2, 1)
        lens = torch.ones(feat.shape[0]).to(feat.device)
        lens = torch.round(lens*feat.shape[1]).int()
        x, masks = self.conformer(feat, lens)
        x = x.permute(0, 2, 1)
        x = self.pooling(x)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.squeeze(1)
        return x

def conformer(feat_dim=80, num_blocks=6, output_size=256, 
        embedding_size=192, input_layer="conv2d", pos_enc_layer_type="rel_pos"):
    model = Conformer(feat_dim=feat_dim, num_blocks=num_blocks, output_size=output_size, 
            embedding_size=embedding_size, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)
    return model




if __name__ == "__main__":
    for i in range(6, 7):
        print("num_blocks is {}".format(i))
        model = conformer(num_blocks=i)

        import time
        model = model.eval()
        time1 = time.time()
        with torch.no_grad():
            for i in range(100):
                data = torch.randn(10, 200, 80)
                embedding = model(data)
                print(embedding.shape)
                print(embedding)
        time2 = time.time()
        val = (time2 - time1)/100
        rtf = val / 5

        total = sum([param.nelement() for param in model.parameters()])
        print("total param: {:.2f}M".format(total/1e6))
        print("RTF {:.4f}".format(rtf))

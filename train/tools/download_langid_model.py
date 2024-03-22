import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")
# Download Thai language sample from Omniglot and cvert to suitable form
signal = language_id.load_audio("speechbrain/lang-id-voxlingua107-ecapa/udhr_th.wav")
prediction =  language_id.classify_batch(signal)
print(prediction)
#  (tensor([[-2.8646e+01, -3.0346e+01, -2.0748e+01, -2.9562e+01, -2.2187e+01,
#         -3.2668e+01, -3.6677e+01, -3.3573e+01, -3.2545e+01, -2.4365e+01,
#         -2.4688e+01, -3.1171e+01, -2.7743e+01, -2.9918e+01, -2.4770e+01,
#         -3.2250e+01, -2.4727e+01, -2.6087e+01, -2.1870e+01, -3.2821e+01,
#         -2.2128e+01, -2.2822e+01, -3.0888e+01, -3.3564e+01, -2.9906e+01,
#         -2.2392e+01, -2.5573e+01, -2.6443e+01, -3.2429e+01, -3.2652e+01,
#         -3.0030e+01, -2.4607e+01, -2.2967e+01, -2.4396e+01, -2.8578e+01,
#         -2.5153e+01, -2.8475e+01, -2.6409e+01, -2.5230e+01, -2.7957e+01,
#         -2.6298e+01, -2.3609e+01, -2.5863e+01, -2.8225e+01, -2.7225e+01,
#         -3.0486e+01, -2.1185e+01, -2.7938e+01, -3.3155e+01, -1.9076e+01,
#         -2.9181e+01, -2.2160e+01, -1.8352e+01, -2.5866e+01, -3.3636e+01,
#         -4.2016e+00, -3.1581e+01, -3.1894e+01, -2.7834e+01, -2.5429e+01,
#         -3.2235e+01, -3.2280e+01, -2.8786e+01, -2.3366e+01, -2.6047e+01,
#         -2.2075e+01, -2.3770e+01, -2.2518e+01, -2.8101e+01, -2.5745e+01,
#         -2.6441e+01, -2.9822e+01, -2.7109e+01, -3.0225e+01, -2.4566e+01,
#         -2.9268e+01, -2.7651e+01, -3.4221e+01, -2.9026e+01, -2.6009e+01,
#         -3.1968e+01, -3.1747e+01, -2.8156e+01, -2.9025e+01, -2.7756e+01,
#         -2.8052e+01, -2.9341e+01, -2.8806e+01, -2.1636e+01, -2.3992e+01,
#         -2.3794e+01, -3.3743e+01, -2.8332e+01, -2.7465e+01, -1.5085e-02,
#         -2.9094e+01, -2.1444e+01, -2.9780e+01, -3.6046e+01, -3.7401e+01,
#         -3.0888e+01, -3.3172e+01, -1.8931e+01, -2.2679e+01, -3.0225e+01,
#         -2.4995e+01, -2.1028e+01]]), tensor([-0.0151]), tensor([94]), ['th'])
# The scores in the prediction[0] tensor can be interpreted as log-likelihoods that
# the given utterance belongs to the given language (i.e., the larger the better)
# The linear-scale likelihood can be retrieved using the following:
print(prediction[1].exp())
#  tensor([0.9850])
# The identified language ISO code is given in prediction[3]
print(prediction[3])
#  ['th: Thai']
  
# Alternatively, use the utterance embedding extractor:
emb =  language_id.encode_batch(signal)
print(emb.shape)
# torch.Size([1, 1, 256])

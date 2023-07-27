import os
import torchaudio
# from speechbrain.pretrained import EncoderClassifier

# classifier = EncoderClassifier.from_hparams(source="nn/GENDER")

# def classify_wav(wav):
#   out_prob, score, index, text_lab = classifier.classify_batch(wav)
#   print(f"out_prob: {out_prob}\tscore: {score}\tindex: {index}\ttext_lab: {text_lab}")
#   result= {
#     "score": float(score.numpy()),
#     "index": int(index.numpy()),
#     "text_lab": text_lab[0]
#   }
#   return result

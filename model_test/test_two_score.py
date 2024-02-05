import argparse
from dguard.interface import PretrainedModel


if __main__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav1", type=str, required=True)
    parser.add_argument("--wav2", type=str, required=True)
    args = parser.parse_args()
    wav1 = args.wav1
    wav2 = args.wav2
    infer1 = PretrainedModel('resnet101_cjsd', mode="compare")
    infer2 = PretrainedModel('resnet221_cjsd_lm', mode="compare")
    infer3 = PretrainedModel('resnet293_cjsd_lm', mode="compare")
    cos_score1, factor1,alpha1 = infer1.inference([wav1,wav2], cmf=False)
    cos_score2, factor2 ,alpha2= infer2.inference([wav1,wav2], cmf=False)
    cos_score3, factor3 ,alpha3= infer3.inference([wav1,wav2], cmf=False)
    cos_score = (cos_score1 + cos_score2 + cos_score3) / 3
    print(f"cos_score: {cos_score}")
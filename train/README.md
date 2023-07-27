## Train SV models

This part is based on the repository [https://github.com/alibaba-damo-academy/3D-Speaker](https://github.com/alibaba-damo-academy/3D-Speaker) and includes the following modifications:

1. Added model DFResNet (56/110/179/233)
2. Added model ResNet-Family
3. Added model ECAPA-TDNN
4. Added model mfa-conformer
5. Added model RepVGG
6. Updated training pipeline to include printing and recording of test data EER and minDCF
7. Added pooling layers
8. Added additional loss functions such as Sphere2

## Training Instructions
Navigate to `egs/voxceleb`
Execute the command `bash run.sh`

Note: The above instructions assume you have all the required dependencies and dataset properly set up.

Enjoy training your models!
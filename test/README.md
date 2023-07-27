# Test_model
#### Test_trials
|-------------|--------------|------------|----------------------------------------------------------------------------|
| Trial_name  | Speaker_nums | pairs_nums |                                                                            |
|-------------|--------------|------------|----------------------------------------------------------------------------|
| VoxCeleb1-O |      40      |    37611   | /datasets/voxceleb1/trials/vox1_O_cleaned.trial                            |
| VoxCeleb1-E |    1251      |   579818   | /datasets/voxceleb1/trials/vox1_E_cleaned.trial                            |
| VoxCeleb1-H |    1251      |   550894   | /datasets/voxceleb1/trials/vox1_H_cleaned.trial                            |
| CTI-10      |     199      |    60000   | /home/duanyibo/dyb/test_model/cti_test/cti.trials                          |
| CNCeleb1    |     200      |  3484292   | /home/duanyibo/dyb/test_model/cnceleb_files/eval/trials/CNC-Eval-Avg.lst   |
| 3Ddevice    |     243      |   180000   | /home/duanyibo/dyb/test_model/3D-speaker/files/trials/trials_cross_device  |
| 3Ddistance  |     243      |   180000   |	/home/duanyibo/dyb/test_model/3D-speaker/files/trials/trials_cross_distance| 
| 3Ddialect   |     243      |   175163   |	/home/duanyibo/dyb/test_model/3D-speaker/files/trials/trials_cross_dialect |
| male        |      53      |    19110   | /datasets/test/testdata_1c_vad_16k/test_trials/male.trials                 |
| female      |      53      |    19900   | /datasets/test/testdata_1c_vad_16k/test_trials/female.trials               |
|-------------|--------------|------------|----------------------------------------------------------------------------|

#### Model_result

[Text for the link](/home/duanyibo/dyb/test_model/Model_Result.md)

#### model

```
CAMPP_EMB_512 
ECAPA_TDNN_1024_EMB_192 
ERES2NET_BASE_EMB_192 
REPVGG_TINY_A0_EMB_512 
DFRESNET56_EMB_512
```

#### 测试集对应scp文件地址

```
#voxceleb1
vox_scp=/home/duanyibo/dyb/test_model/voxceleb1/wav.scp
data_path=/datasets/voxceleb1/dev/wav

#cnceleb1_eval（enroll+test）
cnceleb_scp=/home/duanyibo/dyb/test_model/cnceleb_files/eval/wav.scp
data_path=/home/duanyibo/dyb/test_model/cnceleb/CN-Celeb_wav/eval

#长江数据
cti_scp=/home/duanyibo/dyb/test_model/cti_test/cti.scp
data_path=/datasets/test/cti_test_dataset_16k_envad_bak

#3D-speaker_test数据
speaker_scp=/home/duanyibo/dyb/test_model/3D-speaker/files/wav.scp
data_path=/home/duanyibo/dyb/test_model/3D-speaker/test

#长江男性测试集
male_scp=/datasets/test/testdata_1c_vad_16k/test_scp/male.scp
data_path=/datasets/test/testdata_1c_vad_16k/male/all

#长江女性测试集
female_scp=/datasets/test/testdata_1c_vad_16k/test_scp/female.scp
data_path=/datasets/test/testdata_1c_vad_16k/female/all
```

#### test_trial

```
trials_vox=
	"
	/datasets/voxceleb1/trials/vox1_O_cleaned.trial 		
	/datasets/voxceleb1/trials/vox1_E_cleaned.trial 
	/datasets/voxceleb1/trials/vox1_H_cleaned.trial
	"
trials_cnceleb=
	/home/duanyibo/dyb/test_model/cnceleb_files/eval/trials/CNC-Eval-Avg.lst
trials_cti=
	/home/duanyibo/dyb/test_model/cti_test/cti.trials
trials_3dspeaker=
	"
	/home/duanyibo/dyb/test_model/3D-speaker/files/trials/trials_cross_device 	
	/home/duanyibo/dyb/test_model/3D-speaker/files/trials/trials_cross_distance 
	/home/duanyibo/dyb/test_model/3D-speaker/files/trials/trials_cross_dialect
	"
trials_male=
	/datasets/test/testdata_1c_vad_16k/test_trials/male.trials
trials_female=
	/datasets/test/testdata_1c_vad_16k/test_trials/female.trials
```

#### 脚本使用

```
bash
cd/home/duanyibo/dyb/test_model
#准备数据
bash local/prepare_test_data.sh 
#运行测试脚本
NCCL_P2P_DISABLE=1 bash test.sh
#绘图
python plot.py
#修改参数
	--folder_path
```


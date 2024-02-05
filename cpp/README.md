## CPP UTILS

### File Structure
```shell
.
├── bin
├── include_vad
├── noc_top_src
├── README.md
├── top_src
├── utils
└── vad_src
```
1. bin: binary files
2. include_vad: include files for vad
3. noc_top_src: utils to generate `<top-K-ID>,<top-K-score>` source files.
4. top_src: utils to generate `<top-1-ID>,<top-1-score>` source files.

### VAD
```shell
# build
cd vad_src
bash build.sh
# Test VAD
../bin/vad_file
```

### Top-K
```shell
# build
cd noc_top_src
bash build.sh
# Test Top-K
../bin/noc_top1_multi_model_test
```
# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import sys
import os
import torch
import copy

from_1_to_n_models = {
    "gpt": {
        "wte.weight": 0,
        "attn.c_attn.weight": 30,
        "attn.c_attn.bias": 30,
        "attn.c_proj.weight": 1,
        "mlp.c_fc.weight": 0,
        "mlp.c_fc.bias": 0,
        "mlp.c_proj.weight": 1,
    },
    "opt": {
        "decoder.embed_tokens.weight": 0,
        "self_attn.k_proj.weight": 0,
        "self_attn.k_proj.bias": 0,
        "self_attn.q_proj.weight": 0,
        "self_attn.q_proj.bias": 0,
        "self_attn.v_proj.weight": 0,
        "self_attn.v_proj.bias": 0,

        "self_attn.out_proj.weight": 1,
        "fc1.weight": 0,
        "fc1.bias": 0,
        "fc2.weight": 1,
    },
    "glm": {
        "word_embeddings.weight": 0,
        "attention.query_key_value.weight": 30,
        "attention.query_key_value.bias": 30,
        "attention.dense.weight": 1,
        "mlp.dense_h_to_4h.weight": 0,
        "mlp.dense_h_to_4h.bias": 0,
        "mlp.dense_4h_to_h.weight": 1,
    },
    "t5": {

    },
}

def check_pytorch_model_mp_size(checkpoint: str, target_mp: int):
    """
    check the checkpoints contains the weights for mp_size = target_mp
    """
    assert os.path.isdir(checkpoint)
    filenames = os.listdir(checkpoint)
    filenames = [
        filename for filename in filenames
        if filename.startswith("pytorch_model")
    ]
    if 'pytorch_model.bin' in filenames and target_mp == 1:
        return True
    else:
        filenames.remove('pytorch_model.bin')
    print(
        "check the weight files in {}, the number of mp_size({}) {} num_of_files({})"
        .format(checkpoint, target_mp,
                "=" if target_mp == len(filenames) else "!=", len(filenames)))
    return target_mp == len(filenames)

def change_pytorch_model_mp_from_1_to_n(model_name_brief, checkpoint: str, target_mp: int):
    trans_keys = from_1_to_n_models.get(model_name_brief, None)
    if trans_keys is None:
        print(f"Not support the model_name: {model_name_brief}")
        os._exit(0)

    if check_pytorch_model_mp_size(checkpoint, target_mp):
        return
    assert os.path.isdir(checkpoint)
    filenames = os.listdir(checkpoint)
    filenames = [
        filename for filename in filenames
        if filename.startswith("pytorch_model")
    ]
    if 'pytorch_model.bin' in filenames and target_mp > 1:
        filenames = ['pytorch_model.bin']
    filenames = [os.path.join(checkpoint, x) for x in filenames]

    if target_mp == len(filenames):
        print("MP size keeps the same.")
        exit(0)

    if checkpoint[-1] == '/':
        new_checkpoint = checkpoint[:-1]
    else:
        new_checkpoint = checkpoint
    preserve_keys = [
        "lr_scheduler",
        "skipped_steps",
        "global_steps",
        "global_samples",
        "dp_world_size",
        "iteration",
        "client_lr_scheduler",
        "np_rng_state",
        "random_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "rng_tracker_states",
    ]

    if target_mp > len(filenames):
        print("Increase MP size.")
        assert target_mp % len(filenames) == 0
        ratio = target_mp // len(filenames)
        for i in range(len(filenames)):
            start = ratio * i
            end = ratio * (i + 1)
            d = torch.load(filenames[i], map_location='cpu')
            # if d.get("module", None) is None:
            #     d["module"] = d

            for j in range(start, end):
                d_new = {}
                shift = j - start
                for k, v in d.items():
                    if k != 'module':
                        if k in preserve_keys:
                            d_new[k] = copy.deepcopy(d[k])
                        elif k == "mp_world_size":
                            d_new[k] = target_mp
                        else:
                            d_new[k] = None
                d_new['module'] = {}
                with torch.no_grad():
                    if "module" in d:
                        d = d["module"]

                    for k, v in d.items():
                        assert len(v.shape) < 3
                        flag = 0
                        for keys in trans_keys:
                            if keys in k:
                                flag = 1
                                # find a key to cut
                                dim = trans_keys[keys]

                                if len(v.shape) == 2:
                                    if dim == 30:
                                        part = v.shape[0] // ratio // 3
                                        d_new['module'][k] = torch.cat([
                                            v[shift * part:(shift + 1) *
                                                           part, :].clone(),
                                            v[(shift + ratio) *
                                              part:(shift + 1 + ratio) *
                                                   part, :].clone(),
                                            v[(shift + 2 * ratio) *
                                              part:(shift + 1 + 2 * ratio) *
                                                   part, :].clone()
                                        ], 0)
                                        break

                                    elif dim == 0:
                                        part = v.shape[dim] // ratio
                                        d_new['module'][k] = v[shift *
                                                       part:(shift + 1) *
                                                            part, :].clone()
                                        break

                                    elif dim == 1:
                                        part = v.shape[dim] // ratio
                                        d_new['module'][k] = v[:, shift *
                                                                  part:(shift + 1) *
                                                                       part].clone()
                                        break

                                elif len(v.shape) == 1:
                                    if dim == 30:
                                        part = v.shape[0] // ratio // 3
                                        d_new['module'][k] = torch.cat([
                                            v[shift * part:(shift + 1) *
                                                           part].clone(),
                                            v[(shift + ratio) *
                                              part:(shift + 1 + ratio) *
                                                   part].clone(),
                                            v[(shift + 2 * ratio) *
                                              part:(shift + 1 + 2 * ratio) *
                                                   part].clone()
                                        ], 0)
                                        break

                                    else :
                                        d_new['module'][k] = v[shift * part:(shift + 1) *
                                                            part].clone()
                                        break

                        if flag == 0:
                            d_new['module'][k] = v.clone()


                print("saving mp_size = {:02d} ".format(j))
                filename = os.path.join(new_checkpoint,
                                        "pytorch_model_{:02d}.bin".format(j))
                torch.save(d_new, filename)

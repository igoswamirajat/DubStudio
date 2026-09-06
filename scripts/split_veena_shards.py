import glob
import json
import os
import shutil
import sys
import torch
import safetensors.torch

def main():
    print('=== Starting Veena Resharding ===')
    snapshot_dirs = glob.glob(r'D:\hf_cache\hub\models--maya-research--veena\snapshots\*')
    if not snapshot_dirs:
        print('ERROR: Veena snapshot directory not found!')
        sys.exit(1)

    snap_dir = snapshot_dirs[0]
    blobs_dir = r'D:\hf_cache\hub\models--maya-research--veena\blobs'
    temp_dir = r'C:\temp_veena_shards'
    os.makedirs(temp_dir, exist_ok=True)

    sf1_path = os.path.join(snap_dir, 'model-00001-of-00002.safetensors')
    sf2_path = os.path.join(snap_dir, 'model-00002-of-00002.safetensors')
    index_path = os.path.join(snap_dir, 'model.safetensors.index.json')

    if not os.path.exists(sf1_path):
        print('model-00001-of-00002.safetensors not found (may already be resharded).')
        shards_existing = glob.glob(os.path.join(snap_dir, 'model-0000*-of-00006.safetensors'))
        if len(shards_existing) == 6:
            print('Already resharded to 6 shards! Skipping.')
            return
        sys.exit(1)

    with open(index_path, 'r', encoding='utf-8') as f:
        orig_index = json.load(f)

    new_weight_map = {}

    def read_tensors(file_path):
        with open(file_path, 'rb') as f:
            hlen = int.from_bytes(f.read(8), 'little')
            header = json.loads(f.read(hlen).decode('utf-8'))
            base = 8 + hlen
            tensors_meta = []
            for k, v in header.items():
                if isinstance(v, dict) and 'data_offsets' in v:
                    size = v['data_offsets'][1] - v['data_offsets'][0]
                    tensors_meta.append((k, size, v['shape'], v['dtype'], v['data_offsets']))
            return base, tensors_meta

    print(f'Reading header of {sf1_path}...')
    base1, tensors1 = read_tensors(sf1_path)

    shard_groups_1 = [[] for _ in range(4)]
    cur = 0
    cur_sz = 0
    target = 1.2 * 1024**3

    for t in tensors1:
        if cur < 3 and cur_sz + t[1] > target and len(shard_groups_1[cur]) > 0:
            cur += 1
            cur_sz = 0
        shard_groups_1[cur].append(t)
        cur_sz += t[1]

    with open(sf1_path, 'rb') as f:
        for idx_group, grp in enumerate(shard_groups_1, start=1):
            shard_fname = f'model-{idx_group:05d}-of-00006.safetensors'
            out_path = os.path.join(temp_dir, shard_fname)
            print(f'Writing {shard_fname} ({len(grp)} tensors) to {temp_dir}...')
            tensors_dict = {}
            for name, sz, shape, dtype_str, offsets in grp:
                f.seek(base1 + offsets[0])
                raw = f.read(offsets[1] - offsets[0])
                tensors_dict[name] = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(shape).clone()
                new_weight_map[name] = shard_fname
            safetensors.torch.save_file(tensors_dict, out_path)
            print(f'Verifying {shard_fname}...')
            chk = safetensors.torch.load_file(out_path)
            assert len(chk) == len(grp), f'Mismatch in {shard_fname}'
            del chk
            del tensors_dict

    st1 = os.stat(sf1_path)
    sf1_blob = None
    for b in os.listdir(blobs_dir):
        bp = os.path.join(blobs_dir, b)
        try:
            if os.stat(bp).st_ino == st1.st_ino:
                sf1_blob = bp
                break
        except Exception:
            pass

    print(f'Removing original {sf1_path} and blob {sf1_blob} to free 4.99 GB on D:...')
    os.remove(sf1_path)
    if sf1_blob and os.path.exists(sf1_blob):
        os.remove(sf1_blob)

    for idx_group in range(1, 5):
        shard_fname = f'model-{idx_group:05d}-of-00006.safetensors'
        src = os.path.join(temp_dir, shard_fname)
        dst = os.path.join(snap_dir, shard_fname)
        print(f'Moving {shard_fname} -> {dst}')
        shutil.move(src, dst)

    print(f'Reading header of {sf2_path}...')
    base2, tensors2 = read_tensors(sf2_path)
    shard_groups_2 = [[] for _ in range(2)]
    cur = 0
    cur_sz = 0
    for t in tensors2:
        if cur < 1 and cur_sz + t[1] > target and len(shard_groups_2[cur]) > 0:
            cur += 1
            cur_sz = 0
        shard_groups_2[cur].append(t)
        cur_sz += t[1]

    with open(sf2_path, 'rb') as f:
        for idx_group, grp in enumerate(shard_groups_2, start=5):
            shard_fname = f'model-{idx_group:05d}-of-00006.safetensors'
            out_path = os.path.join(temp_dir, shard_fname)
            print(f'Writing {shard_fname} ({len(grp)} tensors) to {temp_dir}...')
            tensors_dict = {}
            for name, sz, shape, dtype_str, offsets in grp:
                f.seek(base2 + offsets[0])
                raw = f.read(offsets[1] - offsets[0])
                tensors_dict[name] = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(shape).clone()
                new_weight_map[name] = shard_fname
            safetensors.torch.save_file(tensors_dict, out_path)
            print(f'Verifying {shard_fname}...')
            chk = safetensors.torch.load_file(out_path)
            assert len(chk) == len(grp), f'Mismatch in {shard_fname}'
            del chk
            del tensors_dict

    st2 = os.stat(sf2_path)
    sf2_blob = None
    for b in os.listdir(blobs_dir):
        bp = os.path.join(blobs_dir, b)
        try:
            if os.stat(bp).st_ino == st2.st_ino:
                sf2_blob = bp
                break
        except Exception:
            pass

    print(f'Removing original {sf2_path} and blob {sf2_blob} to free 2.45 GB on D:...')
    os.remove(sf2_path)
    if sf2_blob and os.path.exists(sf2_blob):
        os.remove(sf2_blob)

    for idx_group in range(5, 7):
        shard_fname = f'model-{idx_group:05d}-of-00006.safetensors'
        src = os.path.join(temp_dir, shard_fname)
        dst = os.path.join(snap_dir, shard_fname)
        print(f'Moving {shard_fname} -> {dst}')
        shutil.move(src, dst)

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    new_index = {
        'metadata': orig_index.get('metadata', {}),
        'weight_map': new_weight_map,
    }
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(new_index, f, indent=2)

    print('=== Successfully resharded Veena model into 6 shards! ===')
    print(f'Updated index at {index_path} with {len(new_weight_map)} weights mapped.')

if __name__ == '__main__':
    main()

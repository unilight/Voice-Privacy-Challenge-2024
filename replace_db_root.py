#!/usr/bin/env python3

import argparse
import os
from tqdm import tqdm
import shutil

def read_file(path):
    with open(path, "r") as f:
        lines = f.read().splitlines()
    ret = {}
    for line in lines:
        _id, content = line.split(" ")
        ret[_id] = content
    return ret

def get_parser():
    parser = argparse.ArgumentParser(description="replace the root path in a wav.scp")
    parser.add_argument("--wavscp", required=True, type=str, help="original wav.scp")
    parser.add_argument("--old_db_root", required=True, type=str, help="original wav.scp")
    parser.add_argument("--new_db_root", required=True, type=str, help="original wav.scp")
    return parser

def main():
    args = get_parser().parse_args()

    print("Load wav.scp")
    wav_paths = read_file(args.wavscp)

    shutil.move(args.wavscp, args.wavscp + ".tmp")

    with open(args.wavscp, "w") as f:
        for _id, wavpath in tqdm(wav_paths.items()):
            new_wavpath = wavpath.replace(args.old_db_root, args.new_db_root)
            f.write(f"{_id} {new_wavpath}\n")

if __name__ == "__main__":
    main()

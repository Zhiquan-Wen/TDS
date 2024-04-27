import  h5py
import struct
import  numpy as np
import pathlib
from mio import MioWriter
import torch
import  time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_h5', required=True)
    parser.add_argument('--output_path', required=True, help='path to save mio feature')
    args = parser.parse_args()

    # save path
    root = pathlib.Path(args.output_path)

    # input h5 features
    f = h5py.File(args.input_h5, 'r')

    print('finish load h5py !!!')

    features = f['features'][:]

    # if required, please unlock to the mio
    # boxes = f['boxes'][:]  # 4, 36

    h5_ids = f['ids'][:].tolist()

    with MioWriter(root) as m:
        # for i, (box, fea) in enumerate(zip(boxes, features)): # if required, please unlock to the mio
        for i, fea in enumerate(features):
            print(f"{i}/{len(features)}")
            with m.create_collection() as c:
                c.set_meta(struct.pack("<I", h5_ids[i]))
                # c.add_object(box.tobytes()) # if require bounding box, unlock it
                c.add_object(fea.tobytes())


if __name__ == '__main__':
    main()

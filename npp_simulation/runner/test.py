#!/usr/bin/env python

import glob
#
#
# # sps = max([int(i.split(".")[1]) for i in sorted(glob.glob("statepoint.*.h5"))])
# # # return sps[-1] if sps else None
# # print(sps)
# #
#
# def latest_statepoint():
#     sps = sorted(glob.glob("statepoint.*.h5"))
#     sps_max_batch = max([int(i.split(".")[1]) for i in sps])
#     return (sps[-1] if sps else None, sps_max_batch,)
#
# latest_statepoint()

import pickle

files = glob.glob("test_data_log_file-*.pkl")
data = list()

for file in files:
    with open(file, "rb") as f:
        data.append(pickle.load(f))
with open("datatest.txt","a") as f:
    f.write(str(data))

print(data[0])

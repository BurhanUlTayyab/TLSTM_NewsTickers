import utils
import numpy as np
import cv2
import common


test_names, test_inputs, test_targets, test_seq_len = utils.get_data_set('sets/jameel/testimg.txt', 0,15)


print test_names

print len(test_names)

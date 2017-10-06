from __future__ import absolute_import
from __future__ import division
from tensorflow.contrib.learn.python.learn.utils.inspect_checkpoint import print_tensors_in_checkpoint_file
import model
import tensorflow as tf
import utils
import codecs
import os
import editdistance
import time
from utils import decode_sparse_tensor

def detect():
    model_restore = "models/ocr.model-0.959379192885-161999" #give the path of your final model file
    test_filename = "test.txt" #txt file containing the paths of all the test images
    output_folder = "test_outputs3/" #where the outputs will be stored
    factor1 = 35 #think of it as number of test_batches
    factor2 = 41 #think of it as the batch size

    ac=0
    logits, inputs, targets, seq_len,W, b = model.get_train_model()

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)


    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, model_restore)
        #saver.restore(sess, "models2/ocr.model-0.929263617018-35999")
        print("Model restored.")
        a=0
        for x in range(0,factor1):
	    	test_names, test_inputs, test_targets, test_seq_len = utils.get_data_set(test_filename,x*factor2,(x+1)*factor2)
	    	print test_inputs[0].shape

    		feed_dict = {inputs: test_inputs, seq_len: test_seq_len}
    		dd,lp = sess.run([decoded[0],log_prob], feed_dict=feed_dict)
    	    	original_list = decode_sparse_tensor(test_targets)
    	    	detected_list = decode_sparse_tensor(dd)
    		names_list = test_names.tolist()
                print "lp", lp



    		for x,fname_save in enumerate(names_list):
    		    result = detected_list[x]
    		    file = codecs.open(output_folder + os.path.basename(fname_save) + ".rnn.txt","w", "utf-8")
    		    #file.write(''.join(result.tolist()))     if result is numpy
    		    file.write(''.join(result))
    		    file.close()




    		if len(original_list) != (len(detected_list)):
    			print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
    					  " test and detect length desn't match")
    			return
    		print("T/F: original(length) <-------> detectcted(length)")
    		total_ed=0
    		total_len=0
    		for idx, number in enumerate(original_list):

    			detect_number = detected_list[idx]
    			"""if os.path.exists("output/"+names_list[idx] + ".out.txt"):
    			    append_write = 'a' # append if already exists
    			else:
    			    append_write = 'w' # make a new file if not
    			f = codecs.open("output/"+names_list[idx] + ".out.txt",append_write, 'utf-8')
    			f.write("\nDetected: "+''.join(detect_number)+"\n"+"Original: ",''.join(number))
    			f.close()"""

    			ed = editdistance.eval(number,detect_number)

    			ln = len(number)
    			edit_accuracy = (ln - ed) / ln
    			"""if (idx % 10 == 0):
    			    print("Edit: ", ed, "Edit accuracy: ", edit_accuracy,"\n", ''.join(number).encode('utf-8'), "(", len(number), ") <-------> ", ''.join(detect_number).encode('utf-8'), "(", len(detect_number), ")")
                """
                total_ed+=ed
                total_len+=ln

    		accuraccy = (total_len -  total_ed) / total_len
    		print("Test Accuracy:", accuraccy)
    		ac+= accuraccy
    return ac/factor1

if __name__ == '__main__':
	print detect()

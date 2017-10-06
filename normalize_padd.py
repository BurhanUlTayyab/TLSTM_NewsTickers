import os
import pickle
import cv2
import numpy as np

currentDirectory = os.path.join(os.getcwd(),'data')

myFiles = {}; #dict with filePath as key, and scaled_width as value
norm_height = 48


print 'Reading the image shapes . . . .'

for root, dirs, files in os.walk(currentDirectory):
	files = np.sort(files)
	#print sorted(dirs)
	for file in files:
		if file.endswith('.png'):
			filePath = os.path.join(root,file)
			img = cv2.imread(filePath,0)
			height, width = np.float32(img.shape)
			norm_width = (width/height)*norm_height
			myFiles[filePath] = int(norm_width)
			print myFiles
'''			
maxWidth = max(myFiles.values())
print 'Max rescaled width = %d \n' % maxWidth

print 'Rescalling and Padding images . . . .'
count = 0
for fname, scaledWidth in myFiles.iteritems():
	count += 1
	if count % 1000 == 0:	
		print '%d/%d images rescaled' % (count, len(myFiles))
	img1 = cv2.imread(fname,0)
	img_scaled = cv2.resize(img1,(scaledWidth,48))
	img_scaled_flip = cv2.flip(img_scaled,1)
	img_padded = cv2.copyMakeBorder(img_scaled_flip,0,0,0,(maxWidth - scaledWidth),cv2.BORDER_CONSTANT,value=255)
	#print '%s: Original shape = %s, Scaled shape = %s, New Shape = %s' % (fname, img1.shape, img_scaled.shape, img_padded.shape)
	cv2.imwrite(fname,img_padded)	

print '%d/%d images rescaled and padded \n' % (count, len(myFiles))


print 'Creating a pickle file . . . . \n'

dirs = os.listdir(currentDirectory)
dirs = sorted(dirs)

my_dict = {}
len_mydict = 0
for dir in dirs:

		newDirectory = os.path.join(currentDirectory,dir)
		tickers = os.listdir(newDirectory)

		#txtFileName = '%s.txt' % dir
		#file = open(txtFileName, "w")

		for ticker in tickers:
			if ticker.endswith('.png'):
				tickerPath = os.path.join(newDirectory,ticker)
				ticker_name = ticker.rstrip('.png')
				my_dict.update({ticker_name: str(myFiles[tickerPath])})
				#file.write(tickerPath)
				#file.write('\n')

		
		print 'Total number of %s tickers = %d' % (dir, len(my_dict) - len_mydict)
		len_mydict = len(my_dict)
		#file.close()

		#print 'Created file: %s' % txtFileName		

pickleName = 'train_widths.pickle'
pickle.dump(my_dict, open(pickleName, "wb"))  # save it into a file				
print 'Created file: %s \n' % pickleName


'''
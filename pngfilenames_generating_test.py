import os
import glob


#inputFolderName = '/home/bharru/Downloads/rnnlib_source_forge_version/examples/urdu_dataset/data/Urdu_OCR_Dataset/testing_noPos'
#inputFolderName = '/home/bharru/Downloads/rnnlib_source_forge_version/examples/urdu_newstickers/Data'
inputFolderName = os.getcwd()

outputFileName = "test.txt"

fileToFind = '.png'
myfiles = []

for root, dirs, files in os.walk(inputFolderName):
	for file in files:
		#print file
		#print root
		if file.endswith(fileToFind):
			filePath = os.path.join(root,file)
			myfiles.append(os.path.join(root,file))
			

print 'Total No of files = ', len(myfiles)
#print myfiles

##############################WRITING INTO A TXT FILE#######################################
file = open(outputFileName, "w")
for pathname in myfiles:
	file.write(pathname)
	file.write('\n')
file.close()

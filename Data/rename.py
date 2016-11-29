import os,sys
path =  os.getcwd()
filenames = os.listdir(path)
for filename in filenames:
	os.rename(filename, filename.replace("(", ""))
	os.rename(filename, filename.replace(" ", ""))

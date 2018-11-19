import string, sys, csv, time
import urllib.request
import shutil
import os.path
import numpy as np
import pickle

# https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
def progress(count, total, suffix=''):
	bar_len = 60
	filled_len = int(round(bar_len * count / float(total)))
	percents = round(100.0 * count / float(total), 1)
	bar = '=' * filled_len + '-' * (bar_len - filled_len)
	sys.stdout.write('[%s] %s%s -> %s\r' % (bar, percents, '%', suffix))
	sys.stdout.flush()  # As suggested by Rom Ruben

# files to download
__downloads = {
	"class_names" : "https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv",
	"train_data" : "https://datasets.figure-eight.com/figure_eight_datasets/open-images/train-annotations-bbox.csv", 
	"validation_data" : "https://datasets.figure-eight.com/figure_eight_datasets/open-images/validation-annotations-bbox.csv",
}

# class of dictionary
class __pcklLibrary(object):
	def __init__(self, list_of_dict):
		self.list_of_dict = list_of_dict

# saves list of dicts as object to file
def save_dicts(lod):
	with open('dict_data.pkl', 'wb') as output:
		lib = __pcklLibrary(lod)
		try:
			pickle.dump(lib, output, pickle.HIGHEST_PROTOCOL)
		except:
			print("\nCould not save object!")
			exit(0)	
	print("\nPickle complete!")

# retrieves pickled object and returns it
def open_dicts():
	print("\nLoading dicts...")
	time_start = time.process_time()
	lib = None
	try:
		with open('dict_data.pkl', 'rb') as input:
			lib = pickle.load(input)
	except:
		print("\nCould not open object from file!")
		print("\nChecking CSVs")
		check_csvs()
		with open('dict_data.pkl', 'rb') as input:
			lib = pickle.load(input)
		pass
	lib = lib.list_of_dict

	time_stop = time.process_time()
	total_time = time_stop - time_start
	print("Total time in fractional seconds: " + str(total_time))
	return {	'class names': lib[0], 
				'class encodings': lib[1], 
				'train codec': lib[2], 
				'train class': lib[3], 
				'val codec': lib[4], 
				'val class': lib[5]
	}

def count_rows(file_name):
	with open(file_name, newline='') as f:
		return len(f.readlines())

# reads each line of the csv file
def read_csv(file_name):
	length = count_rows(file_name)
	with open(file_name, newline='') as csvfile:
		count = 0
		reader = csv.reader(csvfile, delimiter=',')
		if file_name != "class_names":
			reader.__next__()
		d1 = {}
		d2 = {}
		for row in reader:
			# fills in class_names_dict
			if file_name == "class_names":
#				print("name: " + row[0] + "\nvalue: " + row[1])
				d1[row[0]] = row[1]
				d2[row[1]] = row[0]
			else:
				if row[0] not in d1:
					d1[row[0]] = [row[1:]]
				else:
					d1[row[0]].append(row[1:])
				
				if row[2] not in d2:
					d2[row[2]] = [row[0]]
				else:
					d2[row[2]].append(row[0])
			count+=1
			if count % 10000 == 0:
				progress(count, length, row[0])
		if file_name != "class_names":
			for key in d2.keys():
				d2[key] = list(set(d2[key]))
		progress(count, length, 'complete        ')
		print()
		return d1, d2
				

# download csv file 
def download_csv(file_name, url):
    # Download the file from `url` and save it locally under `file_name`:
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

# checks for csv's, then read them and create dictionaries to be returned
def check_csvs():
	for file in __downloads:
		print("\nFile name: " + file + "\nurl: " + __downloads[file])
		url = __downloads[file]

		# if file does not exist; download it
		if not os.path.exists(file):
			try:
				print("\nDownloading file: " + file)
				download_csv(file, url)
			except IOError:
				print("\nERROR: Could not download file!\n")
				exit(1)

	# file exists
	print("\nReading files" + "     (Might take a while)")
	print("\n\tOnce the pickle file is saved this won't happen again...")
#	name = "validation_data"
	time_start = time.process_time()
	result_dict = []
	for name in __downloads.keys():
		print("Creating dict for: " + name )
		result_dict.extend(read_csv(name))
	time_stop = time.process_time()
	total_time = time_stop - time_start

	print("Total time in fractional seconds: " + str(total_time))

	print("\nPickling...")
	save_dicts(result_dict)
#	print(valid_dict_image["0b44f28fa177010c"])
#	print(valid_dict_class["/m/035r7c"])

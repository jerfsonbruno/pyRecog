import re

def max_min_normalize(all_datapoints, datapoint):
	return ( datapoint - min(all_datapoints) ) / ( max(all_datapoints) - min(all_datapoints) )

class DataSet:

	def __init__(self, matrix, name="DataSet"):
		self.data = matrix
		self.name = name


	def getAttribute(self, index):
		d = []
		for i in range(0, len(self.data)):
			d += [self.data[i][index]]
		return d

	def getNumInstances(self):
		return len(self.data)

	def getNumAttributes(self):
		return len(self.data[0])

	def normalize(self, normFunction=max_min_normalize):
		for i in range(0, len(self.data[0])):
			attribute = self.getAttribute(i)
			if type(attribute[0]) != float:
					continue
			for j in range(0, len(attribute)):
				self.data[j][i] = normFunction(attribute, attribute[j])

	def simb2num(self, columnIndex):
		if type(self.data[0][columnIndex]) != str:
			return
		clss = set()
		for li in range(0, len(self.data)):
			clss.add(self.data[li][columnIndex])
		dic = {}
		i = 0
		for cls in clss:
			dic[cls] = i
			i += 1
		for li in range(0, len(self.data)):
			self.data[li][columnIndex] = dic[self.data[li][columnIndex]]

	@staticmethod
	def readFromArff(arffFileName):
		data_matrix = []
		f = open(arffFileName)
		ds = f.read()
		f.close()
		name = ""
		lines = ""
		try:
			name = re.search("\@relation\s+([\w\d]+)", ds, re.I).group(1)
		except Exception:
			name = "DataSet"
		try:
			lines = ds.split("@data")[1]
			lines = lines.split("\n")
		except Exception:
			print("File does not conform with ARFF file format")
			return
		data = []
		for line in lines:
			if line == "":
				continue
			dataline = []
			attrs = line.split(",")
			for attr in attrs:
				at = ''
				try:
					at = float(attr)
				except ValueError:
					try:
						at = str(attr)
					except ValueError:
						continue
				dataline += [at]
			data += [dataline]
		return DataSet(data, name)

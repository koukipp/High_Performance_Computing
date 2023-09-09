import filecmp

with open('Image_data/texture17695.bin.cluster_centres') as f1:
	with open('Image_data/sequential_2000/texture17695.bin.cluster_centres') as f2:
		for line1, line2 in zip(f1, f2): # read rest of lines
			array1 = [float(x) for x in line1.split()]
			array2 = [float(x) for x in line2.split()]

			for x1, x2 in zip(array1, array2):
				if(abs(x1 - x2) > 0.0010):
					print("Incorrect results")
					exit()

if(not filecmp.cmp('Image_data/texture17695.bin.membership', 'Image_data/sequential_2000/texture17695.bin.membership')):
	print("Incorrect results")
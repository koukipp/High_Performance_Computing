import sys
import subprocess

powers = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

for power in powers:
	for counter in range(12):
			py2output = subprocess.check_output(['./a.out', "16", str(power)])
			times = py2output.decode('ascii').split(" ")
			times[0] = float(times[0])
			times[1] = float(times[1])

			print(times)

	print()
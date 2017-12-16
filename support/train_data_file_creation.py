file = open("C:\\Users\\AshishSajwan\\Documents\\Coursework\\ROB_599\\Project\\Perception\\darknet-master\\darknet-master\\build\\darknet\\x64\\data\\train.txt","w")
n = 6487
for i in range(n):
	file.write("data/obj/" + str(i) + ".jpg" + "\n")
file.close
#script0 = 'simData.py'
repeat = 1
useGradsFlag = True
poly_deg = 2
numObs = 328
numMo = 300
cntry = 'FR'
fixMb = False
script = 'bayesMelding.py'
file = 'submitTasks.sh'
f1 = open(file, 'wb')
f1.write('#!/bin/bash\n\n')
numFolds = 10


for s in range(120, 125):
	for fold in range(numFolds):
		file = 'task' + str(s) + 'rep' + str(repeat) + 'fold' + str(fold) + '.sh'
		f = open(file, 'wb')
		# f.write('#!/bin/bash')
		# f.write('\n\n')
		command = '#!/bin/bash\n\n' + \
		 '#PBS -d .\n' + \
		 '#PBS -e logs\n' + \
	     '#PBS -o logs\n' + \
	     '#PBS -N  seed' + str(s) + 'fold' + str(fold) + '\n' + \
	     '#PBS -l nodes=1:ppn=2,vmem=12gb\n' + \
	     '#PBS -q fat1278q\n' + \
	     '#PBS -W group_list=group2\n' + \
	     '#PBS -m abe\n\n' 
		command = command + 'stdbuf -oL python ' + script +  ' -SEED ' + str(s)  + ' -repeat ' + str(repeat) + \
		  ' -withPrior ' + str(False) + ' -o ' + '\'Output\''  + ' -poly_deg ' + \
		 str(poly_deg)   + ' -cntry ' + '\'' + str(cntry) + '\''  + ' -numMo ' + \
		 str(numMo) + ' -numObs ' + str(numObs) + ' -fixMb ' + str(fixMb) + ' -idxFold ' + str(fold) + \
		 ' > logs/output' + str(s)   + '_repeat' + str(repeat) + '_cntry' + str(cntry) + '_numObs' + str(numObs) + \
		 '_numMo' + str(numMo) + '_fixMb'+ str(fixMb) + '_idxFold' + str(fold)
		f.write(command)
		f.close()
		command0 = 'qsub -V task' + str(s) + 'rep' + str(repeat) + 'fold' + str(fold) + '.sh'
		f1.write(command0)
		f1.write('\n')	
f1.close



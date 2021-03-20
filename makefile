all:
	cls
	python HW_1_CylinderVolume_OneNeuron_Sigmoid.py 100000 0
	python HW_1_CylinderVolume_OneNeuron_Tanh.py 1000 0
	python HW_1_CylinderVolume_TwoNeurons_Sigmoid.py 1000 0
	python HW_1_CylinderVolume_TwoNeurons_Tanh.py 1000 0
	python HW_1_CylinderVolume_ThreeNeurons_Sigmoid.py 1000 0
teacher:
	cls
	python LearningAlgorithm_teacher.py
q2:
	cls
	python HW_1_CylinderVolume_OneNeuron_Sigmoid.py
q3:
	cls
	python HW_1_CylinderVolume_OneNeuron_Tanh.py
q5:
	cls
	python HW_1_CylinderVolume_TwoNeurons_Sigmoid.py
q6:
	cls
	python HW_1_CylinderVolume_TwoNeurons_Tanh.py
q8:
	cls
	python HW_1_CylinderVolume_ThreeNeurons_Sigmoid.py
READ_ME:
	cls
	type READ_ME.txt

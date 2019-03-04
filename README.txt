0. git clone https://github.com/lee1258561/HW2-Randomized-Optimization.git
	The data and model parameters in this repository is sufficient to reproduce the result. No need to download the original data.
1. Setup
	install Java 8 SDK from here http://www.oracle.com/technetwork/java/javase/downloads/index.html
	install Ant http://ant.apache.org/
	run:
		cd ABAGAIL/
		ant # compile the source code

	setup python if you want to recreate the figure from the results that are already upload to github	
	using python 3.7.2
	run:
		pip install numpy, matplotlib

3. Usage (run under ABAGAIL/ dir):
	java -cp ABAGAIL.jar opt.test.MNISTTest [RHC|SA|GA] 
		Run all three optimization algorithms on the HW1 Neural Network for MNIST dataset and log the results. If the algorithm parameter is specified, only that algorithm will be run.

	java -cp ABAGAIL.jar opt.test.TravelingSalesmanTest
		Run and log the result of optimizing Traveling Salesman problem with each of the four algorithms

	java -cp ABAGAIL.jar opt.test.FourPeaksTest
		Run and log the result of optimizing Four Peaks Optimization with each of the four algorithms

	java -cp ABAGAIL.jar opt.test.MaxKColoringTest
		Run and log the result of optimizing Max-K-Coloring problem with each of the four algorithms

	python plot.py
		create all figures presented in the report. Placed under fig/

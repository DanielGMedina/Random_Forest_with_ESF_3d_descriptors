#include <pcl/io/pcd_io.h>
#include <pcl/features/esf.h>

#include <pcl/visualization/histogram_visualizer.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <Eigen/Core>
#include <ctime>
#include <sys/timeb.h>

#include <stdexcept>
#include <random>

#include "andres/marray.hxx"
#include "andres/ml/decision-trees.hxx"

using namespace std;
using namespace boost;

// Function to the get the ESF histograms of all the clusters
int saveHistograms(string fileName, string pathDir){

	// Cloud for storing the pointClouds
	pcl::PointCloud<pcl::PointXYZ>::Ptr globalPointCloud(new pcl::PointCloud<pcl::PointXYZ>);

	DIR *dir;
	struct dirent *ent;
	int numGlobalClusters = 0;

	ofstream descriptors_output;

	// Open the directore where the pointclouds and the clusters are
	if ((dir = opendir (pathDir.c_str())) != NULL) {
		descriptors_output.open(fileName);
	  while ((ent = readdir (dir)) != NULL) {
	  	if(strcmp(ent->d_name,"..") != 0 && strcmp(ent->d_name,".") != 0){
				// Parse to get the txt's files where the clusters of each pointcloud are
				vector<string> elems;
				string item;
				char delim = '.';
				stringstream ss(ent->d_name);
		  	while(getline(ss,item,delim)){
		  		elems.push_back(item);
		  	}

		  	if(elems[1] == "txt"){
		  		string pathFile = pathDir+elems[0] + ".pcd";
		  		if (pcl::io::loadPCDFile<pcl::PointXYZ>(pathFile.c_str(), *globalPointCloud) != 0)
					{
		  				exit(0);
					}

					string data(pathDir+ent->d_name);
					ifstream in(data.c_str());
					if (!in.is_open()) exit(0);

					typedef tokenizer< escaped_list_separator<char> > Tokenizer;
					vector< string > vec;
					string line;

					std::vector<int*> pointIds;
					int numLocalClusters = 0;

					// Load the pointcloud of each cluster
					descriptors_output << fixed;
					descriptors_output.precision(16);

					while (getline(in,line))
					{
						vector<int> id_points;
						pcl::PointCloud<pcl::PointXYZ>::Ptr object(new pcl::PointCloud<pcl::PointXYZ>);
						if(numLocalClusters != 0){
							Tokenizer tok(line);
							vec.assign(tok.begin(),tok.end());

							string point;
							char lim = ' ';
				  		stringstream points(vec[0]);
				  		istringstream points_stream(vec[0]);

							//if(atoi(vec[2].c_str()) == 1){
								for(int i; points_stream >> i;){
					  			id_points.push_back(i);
					  		}

								//classes->insert(std::pair<int,int>(numLocalClusters+numGlobalClusters,atoi(vec[1].c_str())));
								pcl::copyPointCloud(*globalPointCloud,id_points,*object);

								// To get the ESF histogram
								pcl::PointCloud<pcl::ESFSignature640>::Ptr localDescriptor(new pcl::PointCloud<pcl::ESFSignature640>);
								pcl::ESFEstimation<pcl::PointXYZ, pcl::ESFSignature640> esf;
								esf.setInputCloud(object);
								esf.compute(*localDescriptor);

								// Store the ESF histogram of each cluster
								for (size_t i = 0; i < 640; i++) {
									descriptors_output << localDescriptor->points[0].histogram[i] << ",";
								}
								if(atoi(vec[1].c_str()) == 3){
									descriptors_output << 2 << ",";
								}else{
									descriptors_output << vec[1] << ",";
								}
								descriptors_output << vec[2] << ",";
								descriptors_output << vec[3] << endl;
								numLocalClusters++;
							//}
						}else{
							numLocalClusters++;
						}
					}

					numGlobalClusters += numLocalClusters - 1;
	  		}
			}
	  }
		descriptors_output.close();
	  closedir (dir);
	} else {
	  /* could not open directory */
	  perror ("");
	  exit(0);
	}

	return numGlobalClusters;
}

int
main(int argc, char** argv)
{
	typedef double Feature;
	typedef unsigned char Label;
	typedef double Probability;

	bool model, descriptors;

	// Check input parameters:
	//	- folder_point_clouds_training: directory where are all the pointclouds to train the model
	//	- folder_point_clouds_test: directory where are all the pointclouds to test the model
	//	-	model_path: RF model to be loaded
	//	- descriptors_name.csv: descriptors of the pointclouds
	if(argc != 4) {
	  	cout << "Execution:" << endl;
			cout << "\t\"./esf_RF -i <folder_point_clouds_test> <folder_point_clouds_test>\"" << endl;
			cout << "\t\"./esf_RF -d <descriptors_name.csv> <folder_point_clouds_test>\"" << endl;
			cout << "\t\"./esf_RF -m <model_path> <folder_point_clouds_test>\"" << endl;
	  	exit(0);
	}

	if (string(argv[1]).compare("-i") == 0) {
		descriptors = false;
		model = false;
	}else if (string(argv[1]).compare("-d") == 0) {
		descriptors = true;
		model = false;
	}else if (string(argv[1]).compare("-m") == 0) {
		model = true;
		descriptors = false;
	}else {
		cout << "Parameters Error, Execution:" << endl;
		cout << "\t\"./esf_RF -i <folder_point_clouds_test> <folder_point_clouds_test>\"" << endl;
		cout << "\t\"./esf_RF -d <descriptors_name.csv> <folder_point_clouds_test>\"" << endl;
		cout << "\t\"./esf_RF -m <model_path> <folder_point_clouds_test>\"" << endl;
		exit(0);
	}

	clock_t startC, finishC;
	ofstream outfile;
	andres::ml::DecisionForest<Feature, Label, Probability> decisionForest;

	// If there is no model to be loaded it is trained
	if(!model && !descriptors){
		// Get the descriptors and the classes to each cluster
		cout << "Getting descriptors..." << endl;
		startC = clock();
		vector<float*> histograms_training = std::vector<float*>();;
		std::map<int,int> classes;
		string name_descriptors = "descriptors.csv";
		int numHistograms = saveHistograms(name_descriptors,string(argv[2]));
		finishC = clock();
		cout << "Finished! - Time: " << (difftime(finishC, startC))/1000000 << " seconds\n" << endl;

		// Define feature matrix
		const size_t numberOfSamples = numHistograms;
		const size_t numberOfFeatures = 640;
		const size_t shape[] = {numberOfSamples, numberOfFeatures};
		andres::Marray<Feature> features(shape, shape + 2);

		// Define labels and set the class
		andres::Marray<Label> labels(shape, shape + 1);

		ifstream infile;
		infile.open (name_descriptors, ifstream::in);
		char cNum[256];
		float histograms[numberOfSamples];
		if (infile.is_open()) {
			for(size_t sample = 0; sample < numberOfSamples; ++sample){
				for(size_t feature = 0; feature < numberOfFeatures; ++feature)
				{
					infile.getline(cNum, 256, ',');
					features(sample, feature) = atof(cNum);
				}
				infile.getline(cNum, 256, ',');
				labels(sample) = atoi(cNum);
				infile.getline(cNum, 256, ',');
				infile.getline(cNum, 256, '\n');
			}
		}
		infile.close();

		// Learn decision forest
		cout << "Training..." << endl;
		startC = clock();
		const size_t numberOfDecisionTrees = 10;
		decisionForest.learn(features, labels, numberOfDecisionTrees);
		finishC = clock();
		cout << "Finished! - Time: " << (difftime(finishC, startC))/1000000 << " seconds\n" << endl;

		std::stringstream sstream;
		decisionForest.serialize(sstream);

		// Store the RF model
		outfile.open("RF_model.txt");
		outfile << sstream.rdbuf();
		outfile.close();

	} else if(descriptors){
		ifstream in;
		in.open(string(argv[2]),ifstream::in);
		int numHistograms = count(std::istreambuf_iterator<char>(in),
         std::istreambuf_iterator<char>(), '\n');
		in.close();

		// Define feature matrix
		const size_t numberOfSamples = numHistograms;
		const size_t numberOfFeatures = 640;
		const size_t shape[] = {numberOfSamples, numberOfFeatures};
		andres::Marray<Feature> features(shape, shape + 2);

		// Define labels and set the class
		andres::Marray<Label> labels(shape, shape + 1);
		ifstream infile;
		infile.open (string(argv[2]), ifstream::in);
		char cNum[256];
		float histograms[numberOfSamples];

		cout << "Reading csv..." << endl;
		startC = clock();
		if (infile.is_open()) {
			for(size_t sample = 0; sample < numberOfSamples; ++sample){
				for(size_t feature = 0; feature < numberOfFeatures; ++feature)
				{
					infile.getline(cNum, 256, ',');
					features(sample, feature) = atof(cNum);
				}
				infile.getline(cNum, 256, ',');
				labels(sample) = atoi(cNum);
				infile.getline(cNum, 256, ',');
				infile.getline(cNum, 256, '\n');
			}
		}
		finishC = clock();
		cout << "Finished! - Time: " << (difftime(finishC, startC))/1000000 << " seconds\n" << endl;
		infile.close();

		// Learn decision forest
		cout << "Training..." << endl;
		startC = clock();
		const size_t numberOfDecisionTrees = 10;
		decisionForest.learn(features, labels, numberOfDecisionTrees);
		finishC = clock();
		cout << "Finished! - Time: " << (difftime(finishC, startC))/1000000 << " seconds\n" << endl;

		std::stringstream sstream;
		decisionForest.serialize(sstream);

		// Store the RF model
		outfile.open("RF_model.txt");
		outfile << sstream.rdbuf();
		outfile.close();
	} else if(model){
		// Load the RF model
		cout << "Reading model..." << endl;
		startC = clock();
		std::ifstream inputfile( string(argv[2]), std::ifstream::binary );
		if(inputfile){
			std::stringstream sstream;
			sstream << inputfile.rdbuf();
			inputfile.close();

			decisionForest.deserialize(sstream);
		}
		finishC = clock();
		cout << "Finished! - Time: " << (difftime(finishC, startC))/1000000 << " seconds\n" << endl;
	}

	// Test the model
	std::map<int,int> classes_test;

	string name_test_descriptors = "descriptors_test.csv";
	int numHistograms_test = saveHistograms(name_test_descriptors,string(argv[3]));

	// Define test feature matrix
  const size_t numberOfSamples_test = numHistograms_test;
  const size_t numberOfFeatures_test = 640;
  const size_t shape_test[] = {numberOfSamples_test, numberOfFeatures_test};

	ifstream infile2;
	infile2.open (name_test_descriptors, ifstream::in);
	char cNum[256];
	float histograms[numberOfSamples_test];
  andres::Marray<Feature> features_test(shape_test, shape_test + 2);

	if (infile2.is_open()) {
		for(size_t sample_test = 0; sample_test < numberOfSamples_test; ++sample_test){
			for(size_t feature_test = 0; feature_test < numberOfFeatures_test; ++feature_test)
			{
				infile2.getline(cNum, 256, ',');
				features_test(sample_test, feature_test) = atof(cNum);;
			}
			infile2.getline(cNum, 256, ',');
			classes_test.insert(std::pair<int,int>(sample_test,atoi(cNum)));
			infile2.getline(cNum, 256, ',');
			infile2.getline(cNum, 256, '\n');
		}
	}
	infile2.close();

	andres::Marray<Probability> probabilities(shape_test, shape_test + 2);
  decisionForest.predict(features_test, probabilities);

	andres::Marray<Probability> probabilities_2(shape_test, shape_test + 2);
  decisionForest.predict(features_test, probabilities_2);

  size_t cnt = 0;

	// Take the predictions
  for (size_t i = 0; i < numberOfSamples_test; ++i){
		if (fabs(probabilities(i) - probabilities_2(i)) < std::numeric_limits<double>::epsilon()){
			++cnt;
			cout << "Probability: " << probabilities(i) << "- class: " << classes_test[i] << endl;
		}
	}

  if (cnt != numberOfSamples_test)
      throw std::runtime_error("two predictions coincide");
}

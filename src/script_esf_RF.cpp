#include <pcl/io/pcd_io.h>
#include <pcl/features/esf.h>
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
void getHistograms(vector<float*> *histograms, std::map<int,int> *classes, string pathDir){

	// Cloud for storing the pointClouds
	pcl::PointCloud<pcl::PointXYZ>::Ptr globalPointCloud(new pcl::PointCloud<pcl::PointXYZ>);

	DIR *dir;
	struct dirent *ent;
	int numGlobalClusters = 0;

	// Open the directore where the pointclouds and the clusters are
	if ((dir = opendir (pathDir.c_str())) != NULL) {
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
				  		for(int i; points_stream >> i;){
				  			id_points.push_back(i);
				  		}

							classes->insert(std::pair<int,int>(numLocalClusters+numGlobalClusters,atoi(vec[1].c_str())));
							pcl::copyPointCloud(*globalPointCloud,id_points,*object);

							// To get the ESF histogram
							pcl::PointCloud<pcl::ESFSignature640>::Ptr localDescriptor(new pcl::PointCloud<pcl::ESFSignature640>);
							pcl::ESFEstimation<pcl::PointXYZ, pcl::ESFSignature640> esf;
							esf.setInputCloud(object);
							esf.compute(*localDescriptor);

							// Store the ESF histogram of each cluster
							histograms->push_back(localDescriptor->points[0].histogram);
						}
						numLocalClusters++;
					}

					numGlobalClusters += numLocalClusters - 1;
	  		}
			}
	  }
	  closedir (dir);
	} else {
	  /* could not open directory */
	  perror ("");
	  exit(0);
	}
}

int
main(int argc, char** argv)
{
	typedef double Feature;
	typedef unsigned char Label;
	typedef double Probability;

	bool model = false;

	// Check input parameters:
	//	- folder_point_clouds_training: directory where are all the pointclouds to train the model
	//	- folder_point_clouds_test: directory where are all the pointclouds to test the model
	//	-	model_path: RF model to be loaded
	if(argc != 4){
		if(argc != 3) {
		  	cout << "Execution:" << endl;
				cout << "\t\"./esf_RF <folder_point_clouds_training> <folder_point_clouds_test>\"" << endl;
				cout << "\t\"./esf_RF <folder_point_clouds_training> <folder_point_clouds_test> <model_path>\"" << endl;
		  	return -1;
		}
	}
	else{
		model = true;
	}

	clock_t startC, finishC;
	ofstream outfile;
	andres::ml::DecisionForest<Feature, Label, Probability> decisionForest;

	// If there is no model to be loaded it is trained
	if(!model){
		//////////DESCRIPTORS////////////
		// Get the descriptors and the classes to each cluster
		cout << "Getting descriptors..." << endl;
		startC = clock();
		vector<float*> histograms_training;
		std::map<int,int> classes;
		getHistograms(&histograms_training,&classes,string(argv[1]));
		finishC = clock();
		cout << "Finished! - Time: " << (difftime(finishC, startC))/1000000 << " seconds\n" << endl;
		////////////////////////////////

		/////////TRAIN RANDOM FOREST CLASSIFIER/////////
		// Define feature matrix
		const size_t numberOfSamples = histograms_training.size();
		const size_t numberOfFeatures = 640;
		const size_t shape[] = {numberOfSamples, numberOfFeatures};
		andres::Marray<Feature> features(shape, shape + 2);
		for(size_t sample = 0; sample < numberOfSamples; ++sample){
			for(size_t feature = 0; feature < numberOfFeatures; ++feature)
			{
				features(sample, feature) = histograms_training[sample][feature];
			}
		}

		// Define labels and set the class
		andres::Marray<Label> labels(shape, shape + 1);
		for(size_t sample = 0; sample < numberOfSamples; ++sample) {
			labels(sample) = classes[sample];
		}

		// Learn decision forest
		cout << "Training..." << endl;
		startC = clock();
		const size_t numberOfDecisionTrees = 3;
		decisionForest.learn(features, labels, numberOfDecisionTrees);
		finishC = clock();
		cout << "Finished! - Time: " << (difftime(finishC, startC))/1000000 << " seconds\n" << endl;

		std::stringstream sstream;
		decisionForest.serialize(sstream);

		// Store the RF model
		outfile.open("RF_model.txt");
		outfile << sstream.rdbuf();
		outfile.close();
		////////////////////////////////////////////////

	}
	else{

		/////////TRAIN RANDOM FOREST CLASSIFIER/////////
		// Load the RF model
		std::ifstream inputfile( string(argv[3]), std::ifstream::binary );
		if(inputfile){
			std::stringstream sstream;
			sstream << inputfile.rdbuf();
			inputfile.close();

			decisionForest.deserialize(sstream);
		}
		////////////////////////////////////////////////

	}

	// Test the model
	vector<float*> histograms_test;
	std::map<int,int> classes_test;
	getHistograms(&histograms_test,&classes_test,string(argv[2]));

	// Define test feature matrix
  const size_t numberOfSamples_test = histograms_test.size();
  const size_t numberOfFeatures_test = 640;
  const size_t shape_test[] = {numberOfSamples_test, numberOfFeatures_test};
  andres::Marray<Feature> features_test(shape_test, shape_test + 2);
  for(size_t sample_test = 0; sample_test < numberOfSamples_test; ++sample_test){
  	for(size_t feature_test = 0; feature_test < numberOfFeatures_test; ++feature_test){
				features_test(sample_test, feature_test) = histograms_test[sample_test][feature_test];

		}
  }

	andres::Marray<Probability> probabilities(shape_test, shape_test + 2);
  decisionForest.predict(features_test, probabilities);

	andres::Marray<Probability> probabilities_2(shape_test, shape_test + 2);
  decisionForest.predict(features_test, probabilities_2);

  size_t cnt = 0;

	// Take the predictions
  for (size_t i = 0; i < numberOfSamples_test; ++i){
		if (fabs(probabilities(i) - probabilities_2(i)) < std::numeric_limits<double>::epsilon()){
			++cnt;
			cout << "P1: " << probabilities(i) << " - P2: " << probabilities_2(i) << "- class: " << classes_test[i] << endl;
		}
	}

  if (cnt != numberOfSamples_test)
      throw std::runtime_error("two predictions coincide");
}

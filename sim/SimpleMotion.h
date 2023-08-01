//
// Created by hoseok on 11/3/18.
//

#ifndef __MS_SIMPLEMOTION_H__
#define __MS_SIMPLEMOTION_H__

#include <string>
#include <vector>
#include "Eigen/Core"

class SimpleMotion {
public:
	SimpleMotion();
	std::vector<std::pair<int, double>> getPose(double ratio);
	std::vector<int> idx;
	std::vector<double> start, end;
	std::string motionName; // ex) "knee flexion", "hip flexion"
};

#endif 
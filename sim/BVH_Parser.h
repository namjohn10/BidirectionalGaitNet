#ifndef __MS_BVHPARSER_H__
#define __MS_BVHPARSER_H__
#include <Eigen/Core>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <utility>
#include <initializer_list>
#include "dart/dart.hpp"
#include <algorithm>
#include "Character.h"

class BVHJoint
{
public:
	BVHJoint() { mDof = 0; }
	~BVHJoint() {}
	void setOffset(Eigen::Vector3d _offset) { mOffset = _offset; }
	Eigen::Vector3d getOffset() { return mOffset; }
	void setParent(BVHJoint *parent) { mParent = parent; }
	Eigen::Isometry3d getTransform(Eigen::VectorXd _pos);

	bool load(std::ifstream &is);

	std::string getName() { return mName; }
	std::vector<BVHJoint *> &getChildren() { return mChildren; }
	int getDof() { return mDof; }
	int getIdx() { return mIdx; }
	void setIdx(int _idx) { mIdx = _idx; }

private:
	std::string mName;
	std::string mType;

	int mDof;
	int mIdx;

	std::vector<int> mChannels;
	Eigen::Vector3d mOffset;
	BVHJoint *mParent;
	std::vector<BVHJoint *> mChildren;
};

class BVH
{
public:
	BVH(const std::string file);
	~BVH();

	int setSkeleton(BVHJoint *joint, int idx);
	int getNumFrame() { return mNumFrame; }
	Eigen::Isometry3d getTransform(Eigen::VectorXd _pos, std::vector<std::string> _list);
	Eigen::VectorXd getRawPose(int idx) { return mRawMotion[idx]; }
	BVHJoint *getRoot() { return mRoot; }
	void addMotion(Eigen::VectorXd p) { mMotion.push_back(p); }

	Eigen::VectorXd getInterpolation(Eigen::VectorXd q1, Eigen::VectorXd q2, double t);
	Eigen::VectorXd getPose(double phase);
	Eigen::VectorXd getTargetPose(double phase);

	double getMaxTime() { return mNumFrame * mFrameTime; }
	std::string getName() { return filename; }
	double getFrameTime() { return mFrameTime; }

	void setMode(bool _symmetry);

	void setRefMotion(Character *_character, dart::simulation::WorldPtr _world);

	void setHeightCalibration(bool _b) { mHeightCalibration = _b; }
	bool getHeightCalibration() { return mHeightCalibration; }

private:
	std::vector<Eigen::VectorXd> mMotion;

	std::string filename;
	BVHJoint *mRoot;
	std::map<std::string, BVHJoint *> mBVHSkeleton;
	int mDof;
	int mNumFrame;
	double mFrameTime;
	bool mSymmetryMode;
	std::vector<Eigen::VectorXd> mRawMotion;
	Character *mCharacter;

	Eigen::Isometry3d mRootTransform;
	bool mHeightCalibration;
	double mHeightOffset;
	double mXOffset;
};

#endif
#include "BVH_Parser.h"
#include <iostream>
#include <Eigen/Geometry>
#include "dart/dart.hpp"

BVH::BVH(const std::string file)
{
	filename = file;
	std::ifstream is(file);
	std::string buffer;
	if (!is)
	{
		std::cout << "Can't Open BVH File" << std::endl;
		exit(-1);
	}
	while (is >> buffer)
	{
		if (buffer == "HIERARCHY")
		{
			mRoot = new BVHJoint();
			mRoot->load(is);
			mDof = setSkeleton(mRoot, 0);
		}
		else if (buffer == "MOTION")
		{
			is >> buffer;	 // "FRAMES"
			is >> mNumFrame; // num_frames

			is >> buffer; // "FRAME TIMES"
			is >> buffer;
			is >> mFrameTime;

			for (int i = 0; i < mNumFrame; i++)
			{
				Eigen::VectorXd pose(mDof);
				for (int j = 0; j < mDof; j++)
					is >> pose[j];

				mRawMotion.push_back(pose);
			}
			break;
		}
	}
	is.close();
	mXOffset = 0.0;
	mHeightOffset = 0.0;
	mHeightCalibration = false;
}

BVH::~BVH()
{
}

void BVH::setMode(bool _symmetry)
{
	mSymmetryMode = _symmetry;
}

int BVH::setSkeleton(BVHJoint *joint, int idx)
{
	joint->setIdx(idx);
	int dof = joint->getDof();
	mBVHSkeleton.insert(std::make_pair(joint->getName(), joint));
	for (auto j_son : joint->getChildren())
		dof += BVH::setSkeleton(j_son, idx + dof);
	return dof;
}

Eigen::Isometry3d
BVH::
	getTransform(Eigen::VectorXd _pos, std::vector<std::string> _list)
{
	Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
	for (auto j : _list)
	{
		auto bvh_j = mBVHSkeleton[j];
		Eigen::VectorXd j_pos = _pos.segment(bvh_j->getIdx(), bvh_j->getDof());
		result = result * bvh_j->getTransform(j_pos);
	}
	return result;
}

Eigen::VectorXd
BVH::
	getInterpolation(Eigen::VectorXd p1, Eigen::VectorXd p2, double t)
{
	Eigen::VectorXd pos = Eigen::VectorXd::Zero(p1.rows());
	for (const auto jn : mCharacter->getSkeleton()->getJoints())
	{
		int dof = jn->getNumDofs();
		if (dof == 0)
			continue;

		int idx = jn->getIndexInSkeleton(0);
		if (dof == 1)
			pos[idx] = p1[idx] * (1 - t) + p2[idx] * t;
		else if (dof == 3)
		{
			Eigen::Quaterniond q1 = Eigen::Quaterniond(BallJoint::convertToRotation(p1.segment(idx, dof)));
			Eigen::Quaterniond q2 = Eigen::Quaterniond(BallJoint::convertToRotation(p2.segment(idx, dof)));
			Eigen::Quaterniond q = q1.slerp(t, q2);
			pos.segment(idx, dof) = BallJoint::convertToPositions(q.toRotationMatrix());
		}
		else if (dof == 6)
		{
			Eigen::Quaterniond q1 = Eigen::Quaterniond(BallJoint::convertToRotation(p1.segment(idx, 3)));
			Eigen::Quaterniond q2 = Eigen::Quaterniond(BallJoint::convertToRotation(p2.segment(idx, 3)));
			Eigen::Quaterniond q = q1.slerp(t, q2);
			pos.segment(idx, 3) = BallJoint::convertToPositions(q.toRotationMatrix());
			pos.segment(idx + 3, 3) = p1.segment(idx + 3, 3) * (1 - t) + p2.segment(idx + 3, 3) * t;
		}
	}
	return pos;
}

Eigen::VectorXd
BVH::
	getPose(double phase)
{
	if (phase < 0)
		phase += 1.0;

	int idx = phase * mNumFrame;

	double idx_f = phase * mNumFrame - idx;

	Eigen::VectorXd p1 = mMotion[idx % mNumFrame];
	for (int i = 0; i < idx / mNumFrame; i++)
		p1.head(6) = FreeJoint::convertToPositions(mRootTransform * FreeJoint::convertToTransform(p1.head(6)));

	Eigen::VectorXd p2 = mMotion[(idx + 1) % mNumFrame];
	for (int i = 0; i < (idx + 1) / mNumFrame; i++)
		p2.head(6) = FreeJoint::convertToPositions(mRootTransform * FreeJoint::convertToTransform(p2.head(6)));

	// Strict Interpolation
	Eigen::VectorXd pos = getInterpolation(p1, p2, idx_f);

	// Height Calibration
	if (mHeightCalibration)
		pos[4] += mHeightOffset;
	pos[3] -= mXOffset;
	pos[3] *= 0.1;
	return pos;
}

Eigen::VectorXd
BVH::
	getTargetPose(double phase)
{
	Eigen::VectorXd pos = getPose(phase);
	if (mSymmetryMode)
		pos = getInterpolation(pos, mCharacter->getMirrorPosition(getPose(phase + 0.5)), 0.5);
	// pos = (pos + mCharacter->getMirrorPosition(getPose(phase + 0.5))) * 0.5;

	return pos;
}

void BVH::
	setRefMotion(Character *_character, dart::simulation::WorldPtr _world)
{
	mCharacter = _character;
	Eigen::Vector3d root_offset = Eigen::Vector3d::Zero();
	bool is_walk = (getName().find("walk") != std::string::npos);
	for (int i = 0; i < getNumFrame(); i++)
	{
		Eigen::VectorXd pos = Eigen::VectorXd::Zero(mCharacter->getSkeleton()->getNumDofs());
		for (auto m : mCharacter->getBVHMap())
		{

			auto ms = m.second;
			int dof = mCharacter->getSkeleton()->getJoint(m.first)->getNumDofs();
			if (dof == 0)
				continue;
			int idx = mCharacter->getSkeleton()->getJoint(m.first)->getIndexInSkeleton(0);
			auto p = getTransform(getRawPose(i), ms);
			if (dof == 1)
			{
				pos[idx] = Eigen::AngleAxisd(p.linear()).angle();

				pos[idx] *= (m.first.find("ForeArm") != std::string::npos ? 0.6 : 1.0);

				if (m.first.find("Tibia") != std::string::npos && is_walk)
				{
					// pos[idx] *= 1.05;
					pos[idx] -= 0.175;
				}
			}
			else if (dof == 3)
			{

				if ((m.first.find("Torso") != std::string::npos || m.first.find("Spine") != std::string::npos) && is_walk)
					p.linear() *= 0.1;

				pos.segment(idx, dof) = BallJoint::convertToPositions(p.linear());

				if ((m.first.find("Torso") != std::string::npos || m.first.find("Spine") != std::string::npos) && is_walk)
					pos[idx] *= 0.1;

				if ((m.first.find("Femur") != std::string::npos) && is_walk)
				{
					pos[idx + 2] *= 0.4;
				}
				if ((m.first.find("TalusR") != std::string::npos) && is_walk)
				{
					pos[idx + 1] += 0.12;
					pos[idx + 2] = 0.0;
				}
				if ((m.first.find("TalusL") != std::string::npos) && is_walk)
				{
					pos[idx + 1] += -0.12;
					pos[idx + 2] = 0.0;
				}
				if ((m.first.find("ArmR") != std::string::npos) && is_walk)
				{
					pos[idx] *= 1.2;
					pos[idx + 1] *= 1.2;
					pos[idx + 2] = M_PI * 0.42;
				}
				if ((m.first.find("ArmL") != std::string::npos) && is_walk)
				{
					pos[idx] *= 1.2;
					pos[idx + 1] *= 1.2;
					pos[idx + 2] = -M_PI * 0.42;
				}
			}
			else if (dof == 6)
			{
				if (is_walk)
					p.linear() *= 0.1;

				if (i == 0)
				{
					root_offset = p.translation();
					p.translation().setZero();
				}
				else
				{
					p.translation() -= root_offset;
				}
				p.translation() *= 0.01;
				pos.segment(idx, dof) = FreeJoint::convertToPositions(p);
			}
		}
		for (int i = 0; i < pos.rows(); i++)
			pos[i] = dart::math::clip(pos[i], mCharacter->getSkeleton()->getPositionLowerLimit(i), mCharacter->getSkeleton()->getPositionUpperLimit(i));

		addMotion(pos);
	}

	// Projection of Roation Matrix
	mRootTransform = FreeJoint::convertToTransform(mMotion.back().head(6)) * FreeJoint::convertToTransform(mMotion.front().head(6)).inverse();
	{
		Eigen::Matrix3d rot = mRootTransform.linear();
		Eigen::Vector3d x = rot * Eigen::Vector3d::UnitX();
		Eigen::Vector3d z = rot * Eigen::Vector3d::UnitZ();
		x[1] = 0.0;
		x = x.normalized();
		z[1] = 0.0;
		z = z.normalized();
		rot = rot.Identity();
		rot(0, 0) = x[0];
		rot(0, 1) = x[1];
		rot(0, 2) = x[2];

		rot(2, 0) = z[0];
		rot(2, 1) = z[1];
		rot(2, 2) = z[2];

		mRootTransform.linear() = rot;
		mRootTransform.translation()[1] = 0.0;
	}

	{
		mHeightOffset = -mCharacter->getSkeleton()->getRootBodyNode()->getCOM()[1];
		Eigen::VectorXd pos_backup = mCharacter->getSkeleton()->getPositions();
		mCharacter->getSkeleton()->setPositions(getPose(1E-6));
		mCharacter->heightCalibration(_world);
		mHeightOffset += mCharacter->getSkeleton()->getRootBodyNode()->getCOM()[1];
		mCharacter->getSkeleton()->setPositions(pos_backup);
	}

	// XOffset To 0
	{
		mXOffset = 0.0;
		for (int i = 0; i < mMotion.size(); i++)
		{
			if (mSymmetryMode)
				mXOffset += (mMotion[i][3] - mMotion[(i + mMotion.size() / 2) % mMotion.size()][3]) * 0.5;
			else
				mXOffset += mMotion[i][3];
		}
		mXOffset /= mMotion.size();
	}
}

Eigen::Isometry3d
BVHJoint::
	getTransform(Eigen::VectorXd _pos)
{
	Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
	for (int i = 0; i < mDof; i++)
	{
		switch (mChannels[i])
		{
		case 0:
			result.linear() *= Eigen::AngleAxisd(_pos[i] * M_PI / 180.0, Eigen::Vector3d::UnitX()).toRotationMatrix();
			break;
		case 1:
			result.linear() *= Eigen::AngleAxisd(_pos[i] * M_PI / 180.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
			break;
		case 2:
			result.linear() *= Eigen::AngleAxisd(_pos[i] * M_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
			break;
		case 3:
			result.translation()[0] += _pos[i];
			break;
		case 4:
			result.translation()[1] += _pos[i];
			break;
		case 5:
			result.translation()[2] += _pos[i];
			break;
		}
	}
	result.translation() += mOffset;
	return result;
}

bool BVHJoint::
	load(std::ifstream &is)
{
	std::string buffer;
	is >> buffer;

	if (buffer == "}")
		return false;

	mType = buffer;
	is >> mName;

	is >> buffer; // for '{'

	is >> buffer;
	if (buffer != "OFFSET")
	{
		std::cout << "Wrong Format offset" << std::endl;
		exit(-1);
	}
	is >> mOffset[0] >> mOffset[1] >> mOffset[2];
	is >> buffer;

	if (buffer == "}")
		return true;

	else if (buffer != "CHANNELS")
	{
		std::cout << "Wrong Format channels" << std::endl;
		exit(-1);
	}

	is >> mDof;
	for (int i = 0; i < mDof; i++)
	{
		is >> buffer;
		std::transform(buffer.begin(), buffer.end(), buffer.begin(), ::toupper);
		if (buffer == "XROTATION")
			mChannels.push_back(0);
		else if (buffer == "YROTATION")
			mChannels.push_back(1);
		else if (buffer == "ZROTATION")
			mChannels.push_back(2);
		else if (buffer == "XPOSITION")
			mChannels.push_back(3);
		else if (buffer == "YPOSITION")
			mChannels.push_back(4);
		else if (buffer == "ZPOSITION")
			mChannels.push_back(5);
	}

	while (true)
	{
		BVHJoint *child = new BVHJoint();
		child->setParent(this);
		if (!child->load(is))
		{
			delete child;
			break;
		}
		else
			mChildren.push_back(child);
	}
	return true;
}

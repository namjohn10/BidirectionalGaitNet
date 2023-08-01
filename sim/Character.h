#ifndef __MS_CHARACTER_H__
#define __MS_CHARACTER_H__
#include "dart/dart.hpp"
#include "DARTHelper.h"
#include "Muscle.h"
#include "SimpleMotion.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

struct ModifyInfo
{
    ModifyInfo() : ModifyInfo(1.0, 1.0, 1.0, 1.0, 0.0) {}
    ModifyInfo(double lx, double ly, double lz, double sc, double to)
    {
        value[0] = lx; // length x (ratio)
        value[1] = ly; // length y (ratio)
        value[2] = lz; // length z (ratio)
        value[3] = sc; // scale (ration)
        value[4] = to; // torsion (angle)
    };
    double value[5];
    double &operator[](int idx) { return value[idx]; }
    double operator[](int idx) const { return value[idx]; }
};
using BoneInfo = std::tuple<std::string, ModifyInfo>;

enum ActuactorType
{
    tor,
    pd,
    mus,
    mass,
    neuromus
};
struct MuscleTuple
{
    // Eigen::VectorXd dt;
    Eigen::VectorXd JtA_reduced;
    Eigen::VectorXd JtP;
    Eigen::MatrixXd JtA;
};
class Character
{
public:
    Character(std::string path, double defaultKp, double defaultKv, double defaultDamping);
    ~Character();

    dart::dynamics::SkeletonPtr getSkeleton() { return mSkeleton; }
    std::map<std::string, std::vector<std::string>> getBVHMap() { return mBVHMap; }

    // For Mirror Function

    std::vector<std::pair<Joint *, Joint *>> getPairs() { return mPairs; }
    Eigen::VectorXd getMirrorPosition(Eigen::VectorXd pos);

    void setLocalTime(double time) { mLocalTime = time; }
    double getLocalTime() { return mLocalTime; }
    double updateLocalTime(double dtime);

    Eigen::VectorXd getSPDForces(const Eigen::VectorXd &p_desired, const Eigen::VectorXd &ext, int inference_per_sim = 1);

    void setPDTarget(Eigen::VectorXd _pdtarget) { mPDTarget = _pdtarget; }
    Eigen::VectorXd getPDTarget() { return mPDTarget; }

    void setTorque(Eigen::VectorXd _torque) { mTorque = _torque; }
    Eigen::VectorXd getTorque() { return mTorque; }

    void setActivations(Eigen::VectorXd _activation) { mActivations = _activation; }
    Eigen::VectorXd getActivations() { return mActivations; }
    void step();

    void setActuactorType(ActuactorType _act) { mActuactorType = _act; }
    ActuactorType getActuactorType() { return mActuactorType; }

    std::vector<dart::dynamics::BodyNode *> getEndEffectors() { return mEndEffectors; }

    Eigen::VectorXd heightCalibration(dart::simulation::WorldPtr _world, bool isStrict = true);
    std::vector<Eigen::Matrix3d> getBodyNodeTransform() { return mBodyNodeTransform; }
    void setMuscles(const std::string path, bool useVelocityForce = false, bool meshLbsWeight = false);
    const std::vector<Muscle *> &getMuscles() { return mMuscles; }

    void clearLogs();

    const std::vector<Eigen::VectorXd> &getTorqueLogs() { return mTorqueLogs; }
    const std::vector<Eigen::VectorXd> &getActivationLogs() { return mActivationLogs; }
    const std::vector<Eigen::Vector3d> &getCOMLogs() { return mCOMLogs; }
    const std::vector<Eigen::Vector3d> &getHeadVelLogs() { return mHeadVelLogs; }
    const std::vector<Eigen::VectorXd> &getMuscleTorqueLogs() { return mMuscleTorqueLogs; }

    Eigen::VectorXd getMirrorActivation(Eigen::VectorXd _activation);

    // MuscleTuple getMuscleTuple(Eigen::VectorXd dt, bool isMirror = false);
    MuscleTuple getMuscleTuple(bool isMirror = false);

    int getNumMuscles() { return mMuscles.size(); }
    int getNumMuscleRelatedDof() { return mNumMuscleRelatedDof; }

    Eigen::VectorXd addPositions(Eigen::VectorXd pos1, Eigen::VectorXd pos2, bool includeRoot = true);

    // Body Parameter
    double getSkelParamValue(std::string name);
    double getTorsionValue(std::string name);
    void setSkelParam(std::vector<std::pair<std::string, double>> _skel_info, bool doOptimization = false);
    void applySkeletonLength(const std::vector<BoneInfo> &info, bool doOptimization = false);
    void applySkeletonBodyNode(const std::vector<BoneInfo> &info, dart::dynamics::SkeletonPtr skel);
    double calculateMetric(Muscle *stdMuscle, Muscle *rtgMuscle, const std::vector<SimpleMotion *> &simpleMotions, const Eigen::EIGEN_VV_VEC3D &x0);
    double fLengthCurve(double minPhaseDiff, double maxPhaseDiff, double lengthDiff) { return 0.007 * pow(minPhaseDiff, 2) + 0.007 * pow(maxPhaseDiff, 2) + 0.5 * pow(lengthDiff, 2); }
    double fRegularizer(Muscle *rtgMuscle, const Eigen::EIGEN_VV_VEC3D &x0)
    {
        double total = 0;
        for (int i = 1; i + 1 < rtgMuscle->mAnchors.size(); i++)
        {
            Anchor *anchor = rtgMuscle->mAnchors[i];
            for (int j = 0; j < anchor->local_positions.size(); j++)
            {
                total += (anchor->local_positions[j] - x0[i - 1][j]).norm() * anchor->weights[j];
            }
        }
        return total;
    }
    void setSimpleMotion(const std::string &simplemotion, const std::string &jointmap);
    double fShape(Muscle *stdMuscle, Muscle *rtgMuscle);

    // For Drawing
    void updateRefSkelParam(dart::dynamics::SkeletonPtr skel) { applySkeletonBodyNode(mSkelInfos, skel); }
    double getGlobalRatio() { return mGlobalRatio; }
    void setTorqueClipping(bool _torqueClipping) { mTorqueClipping = _torqueClipping; }
    bool getTorqueClipping() { return mTorqueClipping; }

    void setIncludeJtPinSPD(bool _includeJtPinSPD) { mIncludeJtPinSPD = _includeJtPinSPD; }
    bool getIncludeJtPinSPD() { return mIncludeJtPinSPD; }

    Eigen::VectorXd posToSixDof(Eigen::VectorXd pos);
    Eigen::VectorXd sixDofToPos(Eigen::VectorXd raw_pos);

private:
    bool mTorqueClipping;

    dart::dynamics::SkeletonPtr mSkeleton;
    dart::dynamics::SkeletonPtr mRefSkeleton;

    Eigen::VectorXd mKp;
    Eigen::VectorXd mKv;
    Eigen::VectorXd mTorqueWeight;

    std::vector<dart::dynamics::BodyNode *> mEndEffectors;
    std::map<std::string, std::vector<std::string>> mBVHMap;
    std::vector<std::pair<Joint *, Joint *>> mPairs;
    std::vector<Eigen::Matrix3d> mBodyNodeTransform;

    double mLocalTime;

    ActuactorType mActuactorType;
    Eigen::VectorXd mTorque;
    Eigen::VectorXd mPDTarget;

    // Muscle
    std::vector<Muscle *> mMuscles;
    std::vector<Muscle *> mRefMuscles;

    int mNumMuscleRelatedDof;
    Eigen::VectorXd mActivations;

    // Log
    std::vector<Eigen::VectorXd> mTorqueLogs;
    std::vector<Eigen::VectorXd> mActivationLogs;
    std::vector<Eigen::Vector3d> mCOMLogs;
    std::vector<Eigen::Vector3d> mHeadVelLogs;
    std::vector<Eigen::VectorXd> mMuscleTorqueLogs;

    // Skeleton Parameters
    bool mLongOpt;
    std::vector<BoneInfo> mSkelInfos;
    double mGlobalRatio;
    std::map<dart::dynamics::BodyNode *, ModifyInfo> modifyLog;
    std::map<std::string, std::vector<SimpleMotion *>> muscleToSimpleMotions;

    bool mIncludeJtPinSPD;
};
#endif

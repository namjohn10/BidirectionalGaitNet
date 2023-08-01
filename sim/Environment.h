#ifndef __MS_ENVIRONMENT_H__
#define __MS_ENVIRONMENT_H__
#include "dart/dart.hpp"
#include "BVH_Parser.h"
#include "Character.h"
#include "dart/collision/bullet/bullet.hpp"

// Struct Motion (include motion (eigen vectorxd) and param (eigen vectorxd)
struct Motion
{
    std::string name;
    Eigen::VectorXd motion;
    Eigen::VectorXd param;
};

struct param_group
{
    std::vector<std::string> param_names;
    std::vector<int> param_idxs;
    double v;
    std::string name;
    bool is_uniform;
};

struct Network
{
    std::string name; // Actually Path
    py::object joint;
    py::object muscle;

    // Only for cascading learning
    Eigen::VectorXd minV;
    Eigen::VectorXd maxV;
};

enum RewardType
{
    deepmimic,
    gaitnet,
    scadiver
};

enum EOEType
{
    abstime,
    tuple
};

class Environment
{
public:
    Environment();
    ~Environment();

    double getRefStride() { return mRefStride; }
    double getRefCadence() { return mBVHs[0]->getMaxTime(); }
    void initialize(std::string metadata);

    // 시뮬레이션 환경 구성
    void addCharacter(std::string path, double kp, double kv, double damping);
    void addObject(std::string path = nullptr);

    Character *getCharacter(int idx) { return mCharacters[idx]; }
    BVH *getBVH(int idx) { return mBVHs[idx]; }

    void setAction(Eigen::VectorXd _action);

    void step(int _step = 0);
    void reset();

    int isEOE();
    void setRefMotion(BVH *_bvh, Character *_character);

    void updateTargetPosAndVel(bool isInit = false);

    Eigen::VectorXd getTargetPositions() { return mTargetPositions; }
    Eigen::VectorXd getTargetVelocities() { return mTargetVelocities; }

    double getLocalPhase(bool mod_one = false, int character_idx = 0, int bvh_idx = 0) { return (mCharacters[character_idx]->getLocalTime() / (mBVHs[bvh_idx]->getMaxTime() / (mCadence / sqrt(mCharacters[0]->getGlobalRatio())))) - (mod_one ? (int)(mCharacters[character_idx]->getLocalTime() / (mBVHs[bvh_idx]->getMaxTime() / (mCadence / sqrt(mCharacters[0]->getGlobalRatio())))) : 0.0); }

    Eigen::VectorXd getState();
    std::pair<Eigen::VectorXd, Eigen::VectorXd> getProjState(const Eigen::VectorXd minV, const Eigen::VectorXd maxV);

    Eigen::VectorXd getJointState(bool isMirror);

    double getReward();
    bool isActionTime() { return mSimulationConut % (mSimulationHz / mControlHz) == 0; }

    Eigen::VectorXd getAction() { return mAction; }

    int getSimulationHz() { return mSimulationHz; }
    int getControlHz() { return mControlHz; }
    std::string getMetadata() { return mMetadata; }

    bool isMirror(int character_idx = 0) { return mEnforceSymmetry && ((mHardPhaseClipping) ? (getNormalizedPhase() > 0.5) : (getLocalPhase(true) > 0.5)); }

    bool isFall();
    dart::simulation::WorldPtr getWorld() { return mWorld; }

    void setIsRender(bool _b) { isRender = _b; }
    bool getIsRender() { return isRender; }

    std::map<std::string, double> getRewardMap() { return mRewardMap; }
    double getActionScale() { return mActionScale; }

    // Metabolic Reward
    void setIncludeMetabolicReward(bool _includeMetabolicReward) { mIncludeMetabolicReward = _includeMetabolicReward; }
    bool getIncludeMetabolicReward() { return mIncludeMetabolicReward; }
    void setMuscleNetwork(py::object nn)
    {
        if (!mLoadedMuscleNN)
        {
            std::vector<int> child_elem;

            for (int i = 0; i < mPrevNetworks.size(); i++)
            {
                mEdges.push_back(Eigen::Vector2i(i, mPrevNetworks.size()));
                child_elem.push_back(i);
            }
            mChildNetworks.push_back(child_elem);
        }

        mMuscleNN = nn;
        mLoadedMuscleNN = true;
    }
    void setMuscleNetworkWeight(py::object w)
    {
        if (!mLoadedMuscleNN)
        {
            std::vector<int> child_elem;

            for (int i = 0; i < mPrevNetworks.size(); i++)
            {
                mEdges.push_back(Eigen::Vector2i(i, mPrevNetworks.size()));
                child_elem.push_back(i);
            }
            mChildNetworks.push_back(child_elem);
        }
        mMuscleNN.attr("load_state_dict")(w);
        mLoadedMuscleNN = true;
    }

    int getNumAction() { return mAction.rows(); }
    int getNumActuatorAction() { return mNumActuatorAction; }

    MuscleTuple getRandomMuscleTuple() { return mRandomMuscleTuple; }
    Eigen::VectorXd getRandomDesiredTorque() { return mRandomDesiredTorque; }

    Eigen::VectorXd getRandomPrevOut() { return mRandomPrevOut; }
    Eigen::VectorXf getRandomWeight()
    {
        Eigen::VectorXf res = Eigen::VectorXf(1);
        res[0] = (float)mRandomWeight;
        return res;
    }

    bool getUseCascading() { return mUseCascading; }
    bool getUseMuscle() { return mUseMuscle; }
    bool isTwoLevelController() { return mCharacters[0]->getActuactorType() == mass; }

    // get Reward Term
    void updateFootStep(bool isInit = false);
    Eigen::Vector3d getCurrentFootStep() { return mCurrentFoot; }
    Eigen::Vector3d getCurrentTargetFootStep() { return mCurrentTargetFoot; }
    Eigen::Vector3d getNextTargetFootStep() { return mNextTargetFoot; }

    RewardType getRewardType() { return mRewardType; }

    double getMetabolicReward();
    double getStepReward();
    double getAvgVelReward();
    double getLocoPrinReward();
    Eigen::Vector3d getAvgVelocity();
    double getTargetCOMVelocity() { return (mRefStride * mStride * mCharacters[0]->getGlobalRatio()) / (mBVHs[0]->getMaxTime() / (mCadence / sqrt(mCharacters[0]->getGlobalRatio()))); }
    double getNormalizedPhase() { return mGlobalTime / (mBVHs[0]->getMaxTime() / (mCadence / sqrt(mCharacters[0]->getGlobalRatio()))) - (int)(mGlobalTime / (mBVHs[0]->getMaxTime() / (mCadence / sqrt(mCharacters[0]->getGlobalRatio())))); }
    double getWorldPhase() { return mWorldTime / (mBVHs[0]->getMaxTime() / (mCadence / sqrt(mCharacters[0]->getGlobalRatio()))) - (int)(mWorldTime / (mBVHs[0]->getMaxTime() / (mCadence / sqrt(mCharacters[0]->getGlobalRatio())))); }

    // For Parameterization

    // Parameters
    double getCadence() { return mCadence; }
    double getStride() { return mStride; }

    void setParamState(Eigen::VectorXd _param_state, bool onlyMuscle = false, bool doOptimization = false);
    void setNormalizedParamState(Eigen::VectorXd _param_state, bool onlyMuscle = false, bool doOptimization = false);
    Eigen::VectorXd getParamState(bool isMirror = false);
    Eigen::VectorXd getNormalizedParamState(Eigen::VectorXd minV, Eigen::VectorXd maxV, bool isMirror = false)
    {
        Eigen::VectorXd norm_p = getParamState(isMirror);
        for (int i = 0; i < norm_p.rows(); i++)
            norm_p[i] = (norm_p[i] - minV[i]) / (maxV[i] - minV[i]);
        return norm_p;
    }
    const std::vector<std::string> &getParamName() { return mParamName; };
    Eigen::VectorXd getParamMin() { return mParamMin; }
    Eigen::VectorXd getParamMax() { return mParamMax; }
    Eigen::VectorXd getParamDefault() { return mParamDefault; }
    Eigen::VectorXd getParamSample();
    const std::vector<param_group> &getGroupParam() { return mParamGroups; }
    void setGroupParam(Eigen::VectorXd v)
    {
        Eigen::VectorXd sampled_param = mParamMin;
        sampled_param.setOnes();
        int i = 0;
        for (auto &p : mParamGroups)
        {
            p.v = v[i];
            for (auto idx : p.param_idxs)
            {
                double param_w = mParamMax[idx] - mParamMin[idx];
                sampled_param[idx] = mParamMin[idx] + param_w * p.v;
            }
            i++;
        }
        setParamState(sampled_param, false, true);
    }
    int getNumParamState() { return mNumParamState; }
    void updateParamState() { setParamState(getParamSample(), false, true); }
    double getLimitY() { return mLimitY; }

    bool getLearningStd() { return mLearningStd; }
    void setLearningStd(bool learningStd) { mLearningStd = learningStd; }
    void poseOptimiziation(int iter = 100);

    Eigen::Vector2i getIsContact();
    const std::vector<Eigen::Vector2i> &getContactLogs() { return mContactLogs; }

    // For Cascading
    Network loadPrevNetworks(std::string path, bool isFirst); // Neot
    std::pair<Eigen::VectorXd, Eigen::VectorXd> getSpace(std::string metadata);
    std::vector<double> getWeights() { return mWeights; }
    std::vector<double> getDmins() { return mDmins; }
    std::vector<double> getBetas() { return mBetas; }

    void setUseWeights(std::vector<bool> _useWeights)
    {
        for (int i = 0; i < _useWeights.size(); i++)
            mUseWeights[i] = _useWeights[i];
    }
    std::vector<bool> getUseWeights() { return mUseWeights; }

    const std::vector<Eigen::VectorXd> &getDesiredTorqueLogs() { return mDesiredTorqueLogs; }

    Eigen::VectorXd getParamStateFromNormalized(Eigen::VectorXd normalizedParamState)
    {
        Eigen::VectorXd paramState = Eigen::VectorXd::Zero(mNumParamState);
        
        for(int i = 0; i < mNumParamState; i++)
            paramState[i] = normalizedParamState[i] * (mParamMax[i] - mParamMin[i]) + mParamMin[i];
        return paramState;
    }

    Eigen::VectorXd getNormalizedParamStateFromParam(Eigen::VectorXd paramState)
    {
        Eigen::VectorXd norm_p = paramState;
        for (int i = 0; i < norm_p.rows(); i++)
            norm_p[i] = (norm_p[i] - mParamMin[i]) / (mParamMax[i] - mParamMin[i]);
        return norm_p;
    }
    int getNumKnownParam() {return mNumKnownParam;}
    double getGlobalTime() { return mGlobalTime; }

private : 
     bool mPhaseUpdateInContolHz;

    Eigen::VectorXd mTargetPositions;
    Eigen::VectorXd mTargetVelocities;
    double mActionScale;

    // Parameter (General)
    int mSimulationHz, mControlHz;

    // Parameter (Muscle)
    bool mUseMuscle;

    int mInferencePerSim;

    // Simulation
    Eigen::VectorXd mAction;

    dart::simulation::WorldPtr mWorld;
    std::vector<Character *> mCharacters;
    std::vector<dart::dynamics::SkeletonPtr> mObjects;
    std::vector<BVH *> mBVHs;

    // Metadata
    std::string mMetadata;

    // Residual Control
    bool mIsResidual;

    // [Advanced Option]
    bool mIncludeMetabolicReward;

    // Cyclic or Not
    bool mCyclic;

    int mSimulationConut;
    int mHeightCalibration; // 0 : No, 1 : Only avoid collision, 2: Strict
    bool mEnforceSymmetry;

    // Reward Map
    bool isRender;
    std::map<std::string, double> mRewardMap;
    bool mIsStanceLearning;

    // Muscle Learning Tuple
    Eigen::VectorXd mRandomDesiredTorque;
    MuscleTuple mRandomMuscleTuple;
    Eigen::VectorXd mRandomPrevOut;
    double mRandomWeight;

    // Network
    py::object mMuscleNN;

    // Reward Type (Deep Mimic or GaitNet)
    RewardType mRewardType;

    // GaitNet
    double mRefStride;
    double mStride;  // Ratio of Foot stride [default = 1]
    double mCadence; // Ratio of time displacement [default == 1]
    bool mIsLeftLegStance;
    Eigen::Vector3d mNextTargetFoot;
    Eigen::Vector3d mCurrentTargetFoot;
    Eigen::Vector3d mCurrentFoot;

    double mPhaseDisplacement;
    double mPhaseDisplacementScale;
    int mNumActuatorAction;

    double mLimitY; // For EOE

    // Offset for Stance phase at current bvh;
    double mStanceOffset;

    bool mLoadedMuscleNN;
    bool mUseJointState;

    bool mLearningStd;

    // Parameter
    std::vector<param_group> mParamGroups;
    Eigen::VectorXd mParamMin;
    Eigen::VectorXd mParamMax;
    Eigen::VectorXd mParamDefault;
    std::vector<std::string> mParamName;
    std::vector<bool> mSamplingStrategy;
    int mNumParamState;

    // Reward Weight
    double mHeadLinearAccWeight;
    double mHeadRotWeight;
    double mStepWeight;
    double mMetabolicWeight;
    double mAvgVelWeight;

    // Simulation Setting
    bool mSoftPhaseClipping;
    bool mHardPhaseClipping;
    int mPhaseCount;
    int mWorldPhaseCount;

    int mSimulationStep;
    EOEType mEOEType;
    double mGlobalTime;
    double mWorldTime;

    // Pose Optimization
    bool mMusclePoseOptimization;
    int mPoseOptimizationMode;

    // Gait Analysis (only work for render mode)
    std::vector<Eigen::Vector2i> mContactLogs;

    // nFor Cascading
    bool mUseCascading;
    std::vector<Network> mPrevNetworks;

    std::vector<Eigen::Vector2i> mEdges;
    std::vector<std::vector<int>> mChildNetworks;

    std::vector<Eigen::VectorXd> mProjStates;
    std::vector<Eigen::VectorXd> mProjJointStates;

    std::vector<double> mDmins;
    std::vector<double> mWeights;
    std::vector<double> mBetas;

    Eigen::VectorXd mState;
    Eigen::VectorXd mJointState;

    py::object loading_network;

    std::vector<bool> mUseWeights; // Onle For Rendering
    int mHorizon;

    std::vector<Eigen::VectorXd> mDesiredTorqueLogs;

    bool mUseNormalizedParamState;
    int mNumKnownParam;
};
#endif
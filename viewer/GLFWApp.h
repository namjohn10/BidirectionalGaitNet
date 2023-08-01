
// #include <pybind11/numpy.h>
#include "dart/gui/Trackball.hpp"
#include "Environment.h"
#include "GLfunctions.h"
#include "ShapeRenderer.h"
#include <glad/glad.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <examples/imgui_impl_glfw.h>
#include <examples/imgui_impl_opengl3.h>
#include <imgui_internal.h>
#include "C3D_Reader.h"

enum MuscleRenderingType
{
    passiveForce = 0,
    contractileForce,
    activatonLevel,
    contracture,
    weakness
};


class GLFWApp
{
public:
    GLFWApp(int argc, char **argv, bool rendermode = true);
    ~GLFWApp();

    void setEnv(Environment *env, std::string metadata = "../data/env.xml");

    void startLoop();
    void initGL();

    void writeBVH(const dart::dynamics::Joint *jn, std::ofstream &_f, const bool isPos = false); // Pose Or Hierarchy
    void exportBVH(const std::vector<Eigen::VectorXd> &motion, const dart::dynamics::SkeletonPtr &skel);

private:
    py::object mns;
    py::object loading_network;

    void update(bool isSave = false);
    void reset();

    // Drawing Component
    void setCamera();

    void drawSimFrame();
    void drawUIFrame();
    void drawUIDisplay();
    void drawGaitNetDisplay();

    void drawGaitAnalysisDisplay();
    void drawGround(double height);
    void drawCollision();

    void drawSkeleton(const Eigen::VectorXd &pos, const Eigen::Vector4d &color, bool isLineSkeleton = false);
    
    void drawThinSkeleton(const dart::dynamics::SkeletonPtr skelptr);

    void drawSingleBodyNode(const BodyNode *bn, const Eigen::Vector4d &color);
    void drawFootStep();
    void drawPhase(double phase, double normalized_phase);

    void drawShape(const dart::dynamics::Shape *shape, const Eigen::Vector4d &color);

    void drawAxis();
    void drawMuscles(const std::vector<Muscle *> muscles, MuscleRenderingType renderingType = activatonLevel, bool isTransparency = true);

    void drawShadow();

    // Mousing Function
    void mouseMove(double xpos, double ypos);
    void mousePress(int button, int action, int mods);
    void mouseScroll(double xoffset, double yoffset);

    // Keyboard Function
    void keyboardPress(int key, int scancode, int action, int mods);

    // Variable
    bool mRenderMode;
    double mWidth, mHeight;
    bool mRotate, mTranslate, mZooming, mMouseDown;

    GLFWwindow *mWindow;
    Environment *mEnv;

    ShapeRenderer mShapeRenderer;
    bool mDrawOBJ;
    bool mSimulation;

    // Trackball/Camera variables
    dart::gui::Trackball mTrackball;
    double mZoom, mPersp, mMouseX, mMouseY;
    Eigen::Vector3d mTrans, mEye, mUp;
    int mCameraMoving, mFocus;

    // Skeleton for kinematic drawing
    dart::dynamics::SkeletonPtr mMotionSkeleton;

    std::vector<std::string> mNetworkPaths;
    std::vector<Network> mNetworks;

    // Reward Map
    std::vector<std::map<std::string, double>> mRewardBuffer;

    // Rendering Option
    bool mDrawReferenceSkeleton;
    bool mDrawCharacter;
    bool mDrawPDTarget;
    bool mDrawJointSphere;
    bool mDrawFootStep;
    bool mStochasticPolicy;
    bool mDrawEOE;

    MuscleRenderingType mMuscleRenderType;
    int mMuscleRenderTypeInt;

    float mMuscleResolution;

    // Muscle Rendering Option
    std::vector<Muscle *> mSelectedMuscles;
    std::vector<bool> mRelatedDofs;

    // Using Weights
    std::vector<bool> mUseWeights;

    // Screen Record
    bool mScreenRecord;
    int mScreenIdx;

    std::vector<std::string> mFGNList;
    std::vector<std::string> mBGNList;
    std::vector<std::string> mC3DList;

    py::object mFGN;
    std::string mFGNmetadata;
    Eigen::Vector3d mFGNRootOffset;
    int selected_fgn;
    int selected_bgn;
    int selected_c3d;
    bool mDrawFGNSkeleton;

    // Motion Buffer
    std::vector<Eigen::VectorXd> mMotionBuffer;
    std::vector<Eigen::Matrix3d> mJointCalibration;

    // BVH Buffer

    std::vector<Eigen::VectorXd> mC3Dmotion;
    int mC3DCount;

    C3D_Reader* mC3DReader;


    // For GVAE
    py::object mGVAE;
    bool mGVAELoaded;
    std::vector<BoneInfo> mSkelInfosForMotions;
    std::vector<Motion> mMotions;
    std::vector<Motion> mAddedMotions;
    Motion mPredictedMotion;

    int mMotionIdx;
    int mMotionFrameIdx;
    Eigen::Vector3d mMotionRootOffset;

    bool mDrawMotion;
    void drawMotions(Eigen::VectorXd motion, Eigen::VectorXd skel_param, Eigen::Vector3d offset = Eigen::Vector3d(-1.0,0,0), Eigen::Vector4d color = Eigen::Vector4d(0.2,0.2,0.8,0.7)) {
        
        // (1) Set Motion Skeleton
        double global = skel_param[2];
        for(auto& m : mSkelInfosForMotions){
            if(std::get<0>(m).find("Head") == std::string::npos)
            {
                std::get<1>(m).value[0] = global;
                std::get<1>(m).value[1] = global;
                std::get<1>(m).value[2] = global;
            }
        }

        
        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("FemurL")->getIndexInSkeleton()]).value[1] *= skel_param[3];
        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("FemurR")->getIndexInSkeleton()]).value[1] *= skel_param[4];
        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("TibiaL")->getIndexInSkeleton()]).value[1] *= skel_param[5];
        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("TibiaR")->getIndexInSkeleton()]).value[1] *= skel_param[6];
        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("ArmL")->getIndexInSkeleton()]).value[0] *= skel_param[7];
        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("ArmR")->getIndexInSkeleton()]).value[0] *= skel_param[8];
        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("ForeArmL")->getIndexInSkeleton()]).value[0] *= skel_param[9];
        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("ForeArmR")->getIndexInSkeleton()]).value[0] *= skel_param[10];

        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("FemurL")->getIndexInSkeleton()]).value[4] = skel_param[11];
        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("FemurR")->getIndexInSkeleton()]).value[4] = skel_param[12];


        mEnv->getCharacter(0)->applySkeletonBodyNode(mSkelInfosForMotions, mMotionSkeleton);

        int pos_dof = mEnv->getCharacter(0)->posToSixDof(mEnv->getCharacter(0)->getSkeleton()->getPositions()).rows();
        if (motion.rows() != pos_dof * 60) {
            std::cout << "Motion Dimension is not matched" << motion.rows()  << std::endl;
            return;
        }

        // (2) Draw Skeleton according to motion
        Eigen::Vector3d pos = offset;
        glColor4f(color[0], color[1], color[2], color[3]);
        
        for(int i = 0; i < 60; i++)
        {
            Eigen::VectorXd skel_pos = mEnv->getCharacter(0)->sixDofToPos(motion.segment(i*pos_dof, pos_dof));
            pos[0] += motion.segment(i * pos_dof, pos_dof)[6];
            pos[1] = motion.segment(i * pos_dof, pos_dof)[7];
            pos[2] += motion.segment(i * pos_dof, pos_dof)[8];

            skel_pos.segment(3, 3) = pos;
            if (i % 6 == 0)
                drawSkeleton(skel_pos, color);
        }

        mEnv->getCharacter(0)->updateRefSkelParam(mMotionSkeleton);
    }

    std::vector<Eigen::VectorXd> mTestMotion;
    Eigen::Vector3d mC3DCOM;
    bool mRenderConditions;
    bool mRenderC3D;
    

};
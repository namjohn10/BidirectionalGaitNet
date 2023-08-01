// Read C3D Data and extract the gait and skeleton conditions
// This is naive implementation of C3D to BVH conversion, we recommend the usage of other softwars for conversion.

#include "Environment.h"
#include "GLfunctions.h"
#include "Character.h"

#ifndef C3D_READER_H
#define C3D_READER_H

// Marker Structure
struct MocapMarker
{
    std::string name;
    Eigen::Vector3d offset;
    BodyNode *bn;

    Eigen::Vector3d getGlobalPos()
    {
        // Body node size
        Eigen::Vector3d size = (dynamic_cast<const BoxShape *>(bn->getShapeNodesWith<VisualAspect>()[0]->getShape().get()))->getSize();
        Eigen::Vector3d p = Eigen::Vector3d(std::abs(size[0]) * 0.5 * offset[0], std::abs(size[1]) * 0.5 * offset[1], std::abs(size[2]) * 0.5 * offset[2]);
        return bn->getTransform() * p;
    }

    void convertToOffset(Eigen::Vector3d pos)
    {
        Eigen::Vector3d size = (dynamic_cast<const BoxShape *>(bn->getShapeNodesWith<VisualAspect>()[0]->getShape().get()))->getSize();
        Eigen::Vector3d p = bn->getTransform().inverse() * pos;
        offset = Eigen::Vector3d(p[0] / (std::abs(size[0]) * 0.5), p[1] / (std::abs(size[1]) * 0.5), p[2] / (std::abs(size[2]) * 0.5));
    }
};

class C3D_Reader
{
    public:
        C3D_Reader(std::string skel_path, std::string marker_path, Environment *env);
        ~C3D_Reader();

        int getFrameRate() { return mFrameRate; }
        

        std::vector<Eigen::VectorXd> loadC3D(std::string path, double torsionL = 0.0, double torsionR = 0.0, double scale = 1.0, double height = 0.0);
        Eigen::VectorXd getPoseFromC3D(std::vector<Eigen::Vector3d>& _pos);
        // Eigen::VectorXd getPoseFromC3D_2(std::vector<Eigen::Vector3d> _pos);
        SkeletonPtr getBVHSkeleton() { return mBVHSkeleton; }

        // get Original Markers
        std::vector<Eigen::Vector3d> getMarkerPos(int idx) {return mOriginalMarkers[idx];}

        void fitSkeletonToMarker(std::vector<Eigen::Vector3d> init_marker, double torsionL = 0.0, double torsionR = 0.0)
        {
            std::vector<Eigen::Vector3d> ref_markers;
            for(auto m: mMarkerSet)
                ref_markers.push_back(m.getGlobalPos());
            
            // Pelvis size
            int idx = mBVHSkeleton->getBodyNode("Pelvis")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[11] - init_marker[10]).norm() / (ref_markers[11] - ref_markers[10]).norm();

            // FemurR size 
            idx = mBVHSkeleton->getBodyNode("FemurR")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[11] - init_marker[20]).norm() / (ref_markers[11] - ref_markers[20]).norm();  // (init_marker[10] - init_marker[14]).norm() / (ref_markers[10] - ref_markers[14]).norm();

            // TibiaR size 
            idx = mBVHSkeleton->getBodyNode("TibiaR")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[16] - init_marker[14]).norm() / (ref_markers[16] - ref_markers[14]).norm();

            // TalusR size
            idx = mBVHSkeleton->getBodyNode("TalusR")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[18] - init_marker[19]).norm() / (ref_markers[18] - ref_markers[19]).norm();

            // FemurL size
            idx = mBVHSkeleton->getBodyNode("FemurL")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[11] - init_marker[20]).norm() / (ref_markers[11] - ref_markers[20]).norm();

            // TibiaL size
            idx = mBVHSkeleton->getBodyNode("TibiaL")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[20] - init_marker[22]).norm() / (ref_markers[20] - ref_markers[22]).norm();

            // TalusL size
            idx = mBVHSkeleton->getBodyNode("TalusL")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[18] - init_marker[19]).norm() / (ref_markers[18] - ref_markers[19]).norm();
            // (init_marker[23] - init_marker[24]).norm() / (ref_markers[23] - ref_markers[24]).norm();

            // Upper Body
            idx = mBVHSkeleton->getBodyNode("Spine")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

            idx = mBVHSkeleton->getBodyNode("Torso")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

            idx = mBVHSkeleton->getBodyNode("Neck")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

            idx = mBVHSkeleton->getBodyNode("Head")->getIndexInSkeleton();
            // std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

            idx = mBVHSkeleton->getBodyNode("ShoulderR")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

            idx = mBVHSkeleton->getBodyNode("ShoulderL")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

            idx = mBVHSkeleton->getBodyNode("ArmR")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

            idx = mBVHSkeleton->getBodyNode("ArmL")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

            idx = mBVHSkeleton->getBodyNode("ForeArmR")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

            idx = mBVHSkeleton->getBodyNode("ForeArmL")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[7] - init_marker[3]).norm() / (ref_markers[7] - ref_markers[3]).norm();

            idx = mBVHSkeleton->getBodyNode("HandR")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[4] - init_marker[3]).norm() / (ref_markers[4] - ref_markers[3]).norm();

            idx = mBVHSkeleton->getBodyNode("HandL")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[3] = (init_marker[4] - init_marker[3]).norm() / (ref_markers[4] - ref_markers[3]).norm();

            // Torsion
            idx = mBVHSkeleton->getBodyNode("FemurR")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[4] = torsionR;

            idx = mBVHSkeleton->getBodyNode("FemurL")->getIndexInSkeleton();
            std::get<1>(mSkelInfos[idx]).value[4] = torsionL;
            
            femurR_torsion = torsionR;
            femurL_torsion = torsionL;
            
            // 10 , 14
            mEnv->getCharacter(0)->applySkeletonBodyNode(mSkelInfos, mBVHSkeleton);
        }
        const std::vector<MocapMarker>& getMarkerSet() { return mMarkerSet; }
        Motion convertToMotion();
        std::vector<Eigen::VectorXd> mConvertedPos;

    private:
        Environment *mEnv;
        SkeletonPtr mBVHSkeleton;

        std::vector<BoneInfo> mSkelInfos;

        // Marker Set
        std::vector<MocapMarker> mMarkerSet;
        std::vector<Eigen::Vector3d> mRefMarkers;
        std::vector<Eigen::Isometry3d> mRefBnTransformation;

        // std::vector<Eigen::VectorXd> mMarkerPos;
        std::vector<std::vector<Eigen::Vector3d>> mOriginalMarkers;
        int mMarkerIdx;
        int mFrameRate;

        double femurR_torsion;
        double femurL_torsion;

        std::vector<Eigen::VectorXd> mCurrentMotion;
        std::vector<double> mCurrentPhi;

};

#endif


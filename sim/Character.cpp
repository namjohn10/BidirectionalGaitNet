#include "Character.h"

static std::map<std::string, int> skeletonAxis = {
    {"Pelvis", 1},
    {"FemurR", 1},
    {"TibiaR", 1},
    {"TalusR", 2},
    // {"HeelR", 1},
    {"FootThumbR", 2},
    {"FootPinkyR", 2},
    {"FemurL", 1},
    {"TibiaL", 1},
    {"TalusL", 2},
    // {"HeelL", 1},
    {"FootThumbL", 2},
    {"FootPinkyL", 2},
    {"Spine", 1},
    {"Torso", 1},
    {"Neck", 1},
    {"Head", 1},
    {"ShoulderR", 0},
    {"ArmR", 0},
    {"ForeArmR", 0},
    {"HandR", 0},
    {"ShoulderL", 0},
    {"ArmL", 0},
    {"ForeArmL", 0},
    {"HandL", 0},
};
static std::tuple<Eigen::Vector3d, double, double> UnfoldModifyInfo(const ModifyInfo &info)
{
    return std::make_tuple(Eigen::Vector3d(info[0], info[1], info[2]), info[3], info[4]);
}

static Eigen::Isometry3d modifyIsometry3d(const Eigen::Isometry3d &iso, const ModifyInfo &info, int axis, bool rotate = true)
{
    Eigen::Vector3d l;
    double s, t;
    std::tie(l, s, t) = UnfoldModifyInfo(info);
    Eigen::Vector3d translation = iso.translation();
    translation = translation.cwiseProduct(l);
    translation *= s;
    auto tmp = Eigen::Isometry3d(Eigen::Translation3d(translation));
    tmp.linear() = iso.linear();
    Eigen::Vector3d angle_axis = Eigen::Vector3d::Zero();
    angle_axis[axis] = 1;
    if (rotate)
        tmp.rotate(Eigen::AngleAxisd(t, angle_axis));
    return tmp;
}

static void modifyShapeNode(BodyNode *rtgBody, BodyNode *stdBody, const ModifyInfo &info, int axis)
{
    Eigen::Vector3d l;
    double s, t;
    std::tie(l, s, t) = UnfoldModifyInfo(info);
    double la = l[axis], lb = l[(axis + 1) % 3], lc = l[(axis + 2) % 3];

    for (int i = 0; i < rtgBody->getNumShapeNodes(); i++)
    {
        ShapeNode *rtgShape = rtgBody->getShapeNode(i), *stdShape = stdBody->getShapeNode(i);
        ShapePtr newShape;
        if (auto rtg = std::dynamic_pointer_cast<CapsuleShape>(rtgShape->getShape()))
        {
            auto std = std::dynamic_pointer_cast<CapsuleShape>(stdShape->getShape());
            double radius = std->getRadius() * s * (lb + lc) / 2, height = std->getHeight() * s * la;
            newShape = ShapePtr(new CapsuleShape(radius, height));
        }
        else if (auto rtg = std::dynamic_pointer_cast<SphereShape>(rtgShape->getShape()))
        {
            auto std = std::dynamic_pointer_cast<SphereShape>(stdShape->getShape());
            double radius = std->getRadius() * s * (la + lb + lc) / 3;
            newShape = ShapePtr(new SphereShape(radius));
        }
        else if (auto rtg = std::dynamic_pointer_cast<CylinderShape>(rtgShape->getShape()))
        {
            auto std = std::dynamic_pointer_cast<CylinderShape>(stdShape->getShape());
            double radius = std->getRadius() * s * (lb + lc) / 2, height = std->getHeight() * s * la;
            newShape = ShapePtr(new CylinderShape(radius, height));
        }
        else if (std::dynamic_pointer_cast<BoxShape>(rtgShape->getShape()))
        {
            auto std = std::dynamic_pointer_cast<BoxShape>(stdShape->getShape());
            Eigen::Vector3d size = std->getSize() * s;
            size = size.cwiseProduct(l);
            newShape = ShapePtr(new BoxShape(size));
        }
        else if (auto rtg = std::dynamic_pointer_cast<MeshShape>(rtgShape->getShape()))
        {
            auto std = std::dynamic_pointer_cast<MeshShape>(stdShape->getShape());
            Eigen::Vector3d scale = std->getScale();
            scale *= s;
            scale = scale.cwiseProduct(l);
            rtg->setScale(scale);
            Eigen::Isometry3d s = stdShape->getRelativeTransform(), r = modifyIsometry3d(s.inverse(), info, axis, false).inverse();
            rtgShape->setRelativeTransform(r);
            newShape = rtg;
        }
        rtgShape->setShape(newShape);
    }
    ShapePtr shape = rtgBody->getShapeNodesWith<DynamicsAspect>()[0]->getShape();
    double mass = stdBody->getMass() * l[0] * l[1] * l[2] * s * s * s;
    dart::dynamics::Inertia inertia;
    inertia.setMass(mass);
    inertia.setMoment(shape->computeInertia(mass));
    rtgBody->setInertia(inertia);
}

Character::
    Character(std::string path, double defaultKp, double defaultKv, double defaultDamping)
{
    mActuactorType = tor;
    mSkeleton = BuildFromFile(path, defaultDamping);
    mSkeleton->setPositions(Eigen::VectorXd::Zero(mSkeleton->getNumDofs()));

    mTorque = Eigen::VectorXd::Zero(mSkeleton->getNumDofs());
    mPDTarget = Eigen::VectorXd::Zero(mSkeleton->getNumDofs());

    mLocalTime = 0.0;

    mKp = Eigen::VectorXd::Ones(mSkeleton->getNumDofs());
    mKv = Eigen::VectorXd::Ones(mSkeleton->getNumDofs());
    mTorqueWeight = Eigen::VectorXd::Ones(mSkeleton->getNumDofs());

    mActivations = Eigen::VectorXd::Zero(mSkeleton->getNumDofs());

    // Kp, Kv 값에 있는 Joint 에 대해서 그 값을 적용.
    TiXmlDocument doc;
    doc.LoadFile(path.c_str());
    TiXmlElement *skel_elem = doc.FirstChildElement("Skeleton");

    for (TiXmlElement *node = skel_elem->FirstChildElement("Node"); node != nullptr; node = node->NextSiblingElement("Node"))
    {
        if (node->Attribute("endeffector") != nullptr)
        {
            if (std::string(node->Attribute("endeffector")) == "True")
                mEndEffectors.push_back(mSkeleton->getBodyNode(std::string(node->Attribute("name"))));
        }
        TiXmlElement *joint_elem = node->FirstChildElement("Joint");
        int dof = mSkeleton->getJoint(node->Attribute("name"))->getNumDofs();

        if (joint_elem->Attribute("bvh") != nullptr)
        {
            std::string bvh_str = joint_elem->Attribute("bvh");
            auto bvh_list = split_string(bvh_str);
            mBVHMap.insert(std::make_pair(node->Attribute("name"), bvh_list));
        }
        if (dof == 0)
            continue;

        int idx = mSkeleton->getJoint(node->Attribute("name"))->getIndexInSkeleton(0);

        // Set PD Parameter
        if (joint_elem->Attribute("kp") != nullptr)
        {
            mKp.segment(idx, dof) = string_to_vectorXd(joint_elem->Attribute("kp"), dof);
            if (joint_elem->Attribute("kv") != nullptr)
                mKv.segment(idx, dof) = string_to_vectorXd(joint_elem->Attribute("kv"), dof);
            else
                for (int i = 0; i < dof; i++)
                    // mKv[i] = sqrt(2 * mKp[i]);
                    mKv[idx + i] = sqrt(2 * mKp[idx + i]);
        }
        else
        {
            mKp.segment(idx, dof) = Eigen::VectorXd::Ones(dof) * defaultKp;
            mKv.segment(idx, dof) = Eigen::VectorXd::Ones(dof) * defaultKv;
        }

        // Set Torque Weight Parameters
        if (joint_elem->Attribute("weight") != nullptr)
            mTorqueWeight.segment(idx, dof) = string_to_vectorXd(joint_elem->Attribute("weight"), dof);
    }

    mKp.head(6).setZero();
    mKv.head(6).setZero();

    // Make Symmetry Joint Pair (For Symmetry)
    // 설명 : Joint 의 이름 뒤에 'L' 이 있으면 'R' 을 찾아서 Pairing 을 진행// 아무것도 없으면 나 자신과 Pair : from 'L' to 'R'
    for (dart::dynamics::Joint *jn : mSkeleton->getJoints())
    {
        if (jn->getName()[jn->getName().size() - 1] == 'R')
            continue;

        if (jn->getName()[jn->getName().size() - 1] == 'L')
        {
            for (dart::dynamics::Joint *jn_2 : mSkeleton->getJoints())
                if ((jn_2->getName().substr(0, jn_2->getName().size() - 1) == jn->getName().substr(0, jn->getName().size() - 1)) && (jn_2->getName() != jn->getName()))
                {
                    mPairs.push_back(std::make_pair(jn, jn_2));

                    Eigen::AngleAxisd first_axis = Eigen::AngleAxisd(jn->getChildBodyNode()->getTransform().linear());
                    first_axis.axis()[1] *= -1;
                    first_axis.axis()[2] *= -1;

                    Eigen::Matrix3d first_rot = first_axis.toRotationMatrix();
                    Eigen::Matrix3d second_rot = jn_2->getChildBodyNode()->getTransform().linear();

                    mBodyNodeTransform.push_back(first_rot.transpose() * second_rot);

                    break;
                }
        }
        else
        {
            mPairs.push_back(std::make_pair(jn, jn));
            mBodyNodeTransform.push_back(Eigen::Matrix3d::Identity());
        }
    }
    mNumMuscleRelatedDof = 0;
    mGlobalRatio = 1.0;
    mLongOpt = false;
    mTorqueClipping = false;

    mRefSkeleton = BuildFromFile(path, defaultDamping);
    for (auto bn : mSkeleton->getBodyNodes())
    {
        ModifyInfo SkelInfo;
        mSkelInfos.push_back(std::make_pair(bn->getName(), SkelInfo));
    }
    for (BodyNode *bodynode : mSkeleton->getBodyNodes())
        modifyLog[bodynode] = ModifyInfo();

    mIncludeJtPinSPD = false;
}

// Explanatation
// Input : Generalized position of character
// Output : Reflexed Position of the generalized position of character
// Method : Pair 된 Joint 의 Angle 을 반전 시킴

Eigen::VectorXd
Character::
    getMirrorPosition(Eigen::VectorXd pos)
{
    for (auto p : mPairs)
    {
        if (p.first->getNumDofs() == 0)
            continue;
        Eigen::VectorXd pos_first = pos.segment(p.second->getIndexInSkeleton(0), p.second->getNumDofs());
        Eigen::VectorXd pos_second = pos.segment(p.first->getIndexInSkeleton(0), p.first->getNumDofs());
        if (p.first->getNumDofs() == 3)
        {
            pos_first[1] *= -1;
            pos_first[2] *= -1;
            pos_second[1] *= -1;
            pos_second[2] *= -1;
        }
        if (p.first->getNumDofs() == 6)
        {
            pos_first[1] *= -1;
            pos_first[2] *= -1;
            pos_first[3] *= -1;
            pos_second[1] *= -1;
            pos_second[2] *= -1;
            pos_second[3] *= -1;
        }
        pos.segment(p.first->getIndexInSkeleton(0), p.first->getNumDofs()) = pos_first;
        pos.segment(p.second->getIndexInSkeleton(0), p.second->getNumDofs()) = pos_second;
    }
    return pos;
}

double
Character::
    updateLocalTime(double dtime)
{
    mLocalTime += dtime;
    return mLocalTime;
}

Eigen::VectorXd
Character::
    getSPDForces(const Eigen::VectorXd &p_desired, const Eigen::VectorXd &ext, int inference_per_sim)
{
    Eigen::VectorXd q = mSkeleton->getPositions();
    Eigen::VectorXd dq = mSkeleton->getVelocities();
    double dt = mSkeleton->getTimeStep() * inference_per_sim;

    Eigen::MatrixXd M_inv = (mSkeleton->getMassMatrix() + Eigen::MatrixXd(dt * mKv.asDiagonal())).inverse();
    Eigen::VectorXd qdqdt = q + dq * dt;

    Eigen::VectorXd p_diff = -mKp.cwiseProduct(mSkeleton->getPositionDifferences(qdqdt, p_desired));
    Eigen::VectorXd v_diff = -mKv.cwiseProduct(dq);

    Eigen::VectorXd ddq = M_inv * (-mSkeleton->getCoriolisAndGravityForces() + p_diff + v_diff + mSkeleton->getConstraintForces() + ext);
    Eigen::VectorXd tau = p_diff + v_diff - dt * mKv.cwiseProduct(ddq);

    tau.head<6>().setZero();

    return tau;
}

void Character::
    step()
{
    switch (mActuactorType)
    {
    case tor:
        mTorqueLogs.push_back(mTorque);
        mSkeleton->setForces(mTorque);
        break;
    case pd:
        mTorque = getSPDForces(mPDTarget, Eigen::VectorXd::Zero(mSkeleton->getNumDofs()));
        // Cwise Product
        // if (mTorque[10] < -1.0)
        //     mTorque[10] = -1.0;
        mTorqueLogs.push_back(mTorqueWeight.cwiseProduct(mTorque));
        mSkeleton->setForces(mTorque);
        break;
    case mus:
    case mass:
    {
        if (mActuactorType == mus)
            mActivationLogs.push_back(mActivations);
        else
            mActivationLogs.push_back(dart::math::clip(mActivations, Eigen::VectorXd::Zero(mActivations.rows()), Eigen::VectorXd::Ones(mActivations.rows())));
        Eigen::VectorXd muscleTorque = mSkeleton->getExternalForces();
        for (int i = 0; i < mMuscles.size(); i++)
        {
            mMuscles[i]->Update();
            // muscle Activation Clipping
            mMuscles[i]->activation = dart::math::clip(mActivations[i], 0.0, 1.0);
            mMuscles[i]->ApplyForceToBody();
        }
        muscleTorque = mSkeleton->getExternalForces() - muscleTorque;
        mMuscleTorqueLogs.push_back(muscleTorque);
        break;
    }
    default:
        break;
    }
    mCOMLogs.push_back(mSkeleton->getCOM());
    mHeadVelLogs.push_back(mSkeleton->getBodyNode("Head")->getCOMLinearVelocity());
}

// Height Calibration
// Input : World Ptr
// Output : Position of the input skeleton where there is no collision with ground
// Caution :
Eigen::VectorXd Character::heightCalibration(dart::simulation::WorldPtr _world, bool isStrict)
{
    // 위로 올라간것도 아래로 내려와서 맞춤.
    if (isStrict)
    {
        double init_y = mSkeleton->getCOM()[1];
        for (auto bn : mSkeleton->getBodyNodes())
            if (init_y < bn->getCOM()[1])
                init_y = bn->getCOM()[1];
        mSkeleton->setPosition(4, mSkeleton->getPosition(4) - init_y);
    }

    auto collisionGroup = _world->getConstraintSolver()->getCollisionGroup();
    dart::collision::CollisionOption option;
    dart::collision::CollisionResult results;
    bool collision = collisionGroup->collide(option, &results);
    while (collision)
    {
        bool isGround = false;
        for (auto bn : results.getCollidingBodyNodes())
            if (bn->getName() == "ground")
            {
                isGround = true;
                break;
            }
        if (!isGround)
            break;
        Eigen::VectorXd pos = mSkeleton->getPositions();
        // pos[3] = 0;
        pos[4] += 1E-3;
        mSkeleton->setPositions(pos);
        collision = collisionGroup->collide(option, &results);
    }
    return mSkeleton->getPositions();
}

// Muscle
void Character::
    setMuscles(const std::string path, bool useVelocityForce, bool meshLbsWeight)
{
    TiXmlDocument doc;
    if (doc.LoadFile(path.c_str()))
    {
        std::cout << "Can't open file : " << path << std::endl;
        return;
    }
    bool uselegacy = false;
    TiXmlElement *muscledoc = doc.FirstChildElement("Muscle");
    for (TiXmlElement *unit = muscledoc->FirstChildElement("Unit"); unit != nullptr; unit = unit->NextSiblingElement("Unit"))
    {
        std::string name = unit->Attribute("name");
        double f0 = std::stod(unit->Attribute("f0"));
        double lm = std::stod(unit->Attribute("lm"));
        double lt = std::stod(unit->Attribute("lt"));
        double pa = std::stod(unit->Attribute("pen_angle"));
        double lmax = std::stod(unit->Attribute("lmax"));
        double type1_fraction = 0.5;
        if (unit->Attribute("type1_fraction") != nullptr)
            type1_fraction = std::stod(unit->Attribute("type1_fraction"));

        Muscle *muscle_elem = new Muscle(name, f0, lm, lt, pa, lmax, type1_fraction, useVelocityForce);
        Muscle *refmuscle_elem = new Muscle(name, f0, lm, lt, pa, lmax, type1_fraction, useVelocityForce);

        bool isValid = true;
        int num_waypoints = 0;

        // For checking Insertion/Origin point of muscles
        BodyNode *origin = mSkeleton->getBodyNode(unit->FirstChildElement("Waypoint")->Attribute("body"));
        BodyNode *insertion;
        bool isLegMuscle = false;
        for (TiXmlElement *waypoint = unit->FirstChildElement("Waypoint"); waypoint != nullptr; waypoint = waypoint->NextSiblingElement("Waypoint"))
        {
            num_waypoints++;
            insertion = mSkeleton->getBodyNode(waypoint->Attribute("body"));
            if (mSkeleton->getBodyNode(waypoint->Attribute("body")) != nullptr)
                if (mSkeleton->getBodyNode(waypoint->Attribute("body"))->getName().find("Talus") != std::string::npos) // || mSkeleton->getBodyNode(waypoint->Attribute("body"))->getName().find("Tibia") != std::string::npos)
                    isLegMuscle = true;
        }

        // Mesh Lbs Weight Skinning is only applied to foot muscles
        // uselegacy = true;
        int i = 0;
        for (TiXmlElement *waypoint = unit->FirstChildElement("Waypoint"); waypoint != nullptr; waypoint = waypoint->NextSiblingElement("Waypoint"))
        {
            std::string body = waypoint->Attribute("body");
            Eigen::Vector3d glob_pos = string_to_vector3d(waypoint->Attribute("p"));
            if (mSkeleton->getBodyNode(body) == NULL)
            {
                isValid = false;
                break;
            }

            if (i == 0 || i == num_waypoints - 1) // one of the insertion/origin points
            {
                muscle_elem->AddAnchor(mSkeleton->getBodyNode(body), glob_pos);
                refmuscle_elem->AddAnchor(mRefSkeleton->getBodyNode(body), glob_pos);
            }
            else
            {
                // if (meshLbsWeight && isLegMuscle)
                //     std::cout << "muscle name : " << name << std::endl;
                muscle_elem->AddAnchor(mSkeleton, mSkeleton->getBodyNode(body), glob_pos, 2, meshLbsWeight && isLegMuscle);
                refmuscle_elem->AddAnchor(mRefSkeleton, mRefSkeleton->getBodyNode(body), glob_pos, 2, meshLbsWeight && isLegMuscle);
            }
            i++;
        }
        if (isValid)
        {
            muscle_elem->SetMuscle();
            if (muscle_elem->GetNumRelatedDofs() > 0)
            {
                mMuscles.push_back(muscle_elem);
                refmuscle_elem->SetMuscle();
                mRefMuscles.push_back(refmuscle_elem);
            }
        }
    }

    // Attach Symmetry pattern
    for (int i = 0; i < mMuscles.size(); i += 2)
    {
        for (int idx = 0; idx < mMuscles[i]->GetAnchors().size(); idx++)
            for (int b_idx = 0; b_idx < mMuscles[i]->GetAnchors()[idx]->num_related_bodies; b_idx++)
            {
                mMuscles[i]->GetAnchors()[idx]->weights[b_idx] = mMuscles[i + 1]->GetAnchors()[idx]->weights[b_idx];
                mRefMuscles[i]->GetAnchors()[idx]->weights[b_idx] = mRefMuscles[i + 1]->GetAnchors()[idx]->weights[b_idx];
                auto name = mMuscles[i]->GetAnchors()[idx]->bodynodes[b_idx]->getName();
                if (mMuscles[i + 1]->GetAnchors()[idx]->bodynodes[b_idx]->getName().substr(0, name.length() - 1) != name.substr(0, name.length() - 1))
                {
                    std::cout << "[DEBUG] Body Node Setting Calibrate : " << mMuscles[i]->name << " " << mMuscles[i + 1]->name << std::endl;
                    exit(-1);
                }
            }
    }

    for (auto m : mMuscles)
        mNumMuscleRelatedDof += m->num_related_dofs;

    mActivations = Eigen::VectorXd::Zero(mMuscles.size());
}

void Character::
    clearLogs()
{
    mTorqueLogs.clear();
    mActivationLogs.clear();
    mCOMLogs.clear();
    mHeadVelLogs.clear();
    mMuscleTorqueLogs.clear();
}

Eigen::VectorXd Character::
    getMirrorActivation(Eigen::VectorXd _activation)
{
    Eigen::VectorXd mirrored_activations = _activation;
    for (int i = 0; i < _activation.rows(); i += 2)
    {
        mirrored_activations[i] = _activation[i + 1];
        mirrored_activations[i + 1] = _activation[i];
    }
    return mirrored_activations;
}

MuscleTuple Character::
    getMuscleTuple(bool isMirror)
{
    if (mMuscles.size() == 0)
    {
        std::cout << "Character::getMuscleTuple()\tNo Muscles" << std::endl;
        exit(-1);
    }
    MuscleTuple mt;

    mt.JtA_reduced = Eigen::VectorXd::Zero(mNumMuscleRelatedDof);

    int n = mSkeleton->getNumDofs();
    int m = mMuscles.size();
    int root_dof = mSkeleton->getRootJoint()->getNumDofs();

    Eigen::VectorXd JtP = Eigen::VectorXd::Zero(n);
    Eigen::MatrixXd JtA = Eigen::MatrixXd::Zero(n, m);

    int i = 0;
    int idx = 0;
    for (auto m : mMuscles)
    {
        m->Update();
        m->related_vec.setZero();
        Eigen::MatrixXd Jt_reduced = m->GetReducedJacobianTranspose();
        auto Ap = m->GetForceJacobianAndPassive();
        Eigen::VectorXd JtA_reduced = Jt_reduced * Ap.first;
        Eigen::VectorXd JtP_reduced = Jt_reduced * Ap.second;

        for (int j = 0; j < m->GetNumRelatedDofs(); j++)
        {
            JtP[m->related_dof_indices[j]] += JtP_reduced[j];
            JtA(m->related_dof_indices[j], i) = JtA_reduced[j];
            m->related_vec[m->related_dof_indices[j]] = JtA_reduced[j];
        }

        mt.JtA_reduced.segment(idx, JtA_reduced.rows()) = JtA_reduced;
        idx += JtA_reduced.rows();
        i++;
    }

    if (isMirror)
    {
        // dt = getMirrorPosition(dt);
        for (int i = 0; i < mMuscles.size(); i += 2)
        {
            Eigen::VectorXd tmp = JtA.col(i);
            JtA.col(i) = getMirrorPosition(JtA.col(i + 1));
            JtA.col(i + 1) = getMirrorPosition(tmp);
        }
        int i = 0;
        int idx = 0;
        for (auto m : mMuscles)
        {
            Eigen::VectorXd JtA_reduced = Eigen::VectorXd::Ones(m->GetNumRelatedDofs());
            for (int j = 0; j < m->GetNumRelatedDofs(); j++)
                JtA_reduced[j] = JtA(m->related_dof_indices[j], i);

            mt.JtA_reduced.segment(idx, JtA_reduced.rows()) = JtA_reduced;
            idx += JtA_reduced.rows();
            i++;
        }
        JtP = getMirrorPosition(JtP);
    }

    // mt.dt = dt.tail(dt.rows() - mSkeleton->getRootJoint()->getNumDofs());
    mt.JtP = JtP.tail(mSkeleton->getNumDofs() - mSkeleton->getRootJoint()->getNumDofs());
    mt.JtA = JtA.block(mSkeleton->getRootJoint()->getNumDofs(), 0, JtA.rows() - mSkeleton->getRootJoint()->getNumDofs(), JtA.cols());

    if (mTorqueClipping)
    {
        std::cout << "Torque Clipping is Deprecated" << std::endl;
        exit(-1);
        // Eigen::VectorXd min_tau = Eigen::VectorXd::Zero(mSkeleton->getNumDofs() - mSkeleton->getRootJoint()->getNumDofs());
        // Eigen::VectorXd max_tau = Eigen::VectorXd::Zero(mSkeleton->getNumDofs() - mSkeleton->getRootJoint()->getNumDofs());
        // for (int i = 0; i < mt.JtA.rows(); i++)
        // {
        //     for (int j = 0; j < mt.JtA.cols(); j++)
        //     {
        //         if (mt.JtA(i, j) < 0)
        //             min_tau[i] += mt.JtA(i, j);
        //         else
        //             max_tau[i] += mt.JtA(i, j);
        //     }
        //     mt.dt[i] = dart::math::clip(mt.dt[i], min_tau[i] + mt.JtP[i], max_tau[i] + mt.JtP[i]);
        // }
    }

    // Test For Reduced Jacobian
    // for (auto m : mMuscles)
    // {
    //     Eigen::VectorXd related_vec_backup = m->related_vec;
    //     m->Update();
    //     Eigen::MatrixXd Jt = m->GetJacobianTranspose();
    //     auto Ap = m->GetForceJacobianAndPassive();
    //     Eigen::VectorXd JtA = Jt * Ap.first;

    //     m->related_vec.setZero();

    //     for (int i = 0; i < JtA.rows(); i++)
    //         if (JtA[i] > 1E-6)
    //             m->related_vec[i] = 1;
    //         else if (JtA[i] < -1E-6)
    //             m->related_vec[i] = -1;

    //     if ((related_vec_backup - m->related_vec).norm() < 1E-6)
    //         std::cout << "DIFFERENT MUSCLE " << m->name << std::endl;
    // }
    return mt;
}

Eigen::VectorXd
Character::addPositions(Eigen::VectorXd pos1, Eigen::VectorXd pos2, bool includeRoot)
{
    int idx_offset = (includeRoot ? 0 : mSkeleton->getRootJoint()->getNumDofs());
    for (auto jn : mSkeleton->getJoints())
    {
        int dof = jn->getNumDofs();
        if (dof == 0)
            continue;
        int idx = jn->getIndexInSkeleton(0) - idx_offset;
        if (dof == 1)
            pos1[idx] += pos2[idx];
        else if (dof == 3)
            // pos1.segment(idx, dof) = pos1.segment(idx, dof) + pos2.segment(idx, dof); // BallJoint::convertToPositions(BallJoint::convertToRotation(pos1.segment(idx, dof)) * BallJoint::convertToRotation(pos2.segment(idx, dof)));
            pos1.segment(idx, dof) = BallJoint::convertToPositions(BallJoint::convertToRotation(pos1.segment(idx, dof)) * BallJoint::convertToRotation(pos2.segment(idx, dof)));
        else if (dof == 6 && includeRoot)
            // pos1.segment(idx, dof) = pos1.segment(idx, dof) + pos2.segment(idx, dof); // FreeJoint::convertToPositions(FreeJoint::convertToTransform(pos1.segment(idx, dof)) * FreeJoint::convertToTransform(pos2.segment(idx, dof)));
            pos1.segment(idx, dof) = FreeJoint::convertToPositions(FreeJoint::convertToTransform(pos1.segment(idx, dof)) * FreeJoint::convertToTransform(pos2.segment(idx, dof)));
    }
    return pos1;
}

void Character::
    setSkelParam(std::vector<std::pair<std::string, double>> _skel_info, bool doOptimization)
{

    // doOptimization = true;
    // Global Setting
    for (auto s : _skel_info)
    {
        if (s.first == "global")
        {
            mGlobalRatio = s.second;
            break;
        }
    }

    for (int i = 0; i < mSkelInfos.size(); i++)
        if (std::get<0>(mSkelInfos[i]).find("Head") == std::string::npos)
        {
            std::get<1>(mSkelInfos[i]).value[0] = mGlobalRatio;
            std::get<1>(mSkelInfos[i]).value[1] = mGlobalRatio;
            std::get<1>(mSkelInfos[i]).value[2] = mGlobalRatio;
        }
    // std::get<1>(mSkelInfos[i]).value[3] = mGlobalRatio;

    for (auto s : _skel_info)
    {
        for (int i = 0; i < mSkelInfos.size(); i++)
        {

            if (std::get<0>(mSkelInfos[i]) == s.first)
            {

                if (s.first.find("Arm") != std::string::npos)
                    std::get<1>(mSkelInfos[i]).value[0] = s.second * mGlobalRatio;
                else
                    std::get<1>(mSkelInfos[i]).value[1] = s.second * mGlobalRatio;
            }

            if (s.first.find("torsion") != std::string::npos)
            {
                if (s.first.find(std::get<0>(mSkelInfos[i])) != std::string::npos)
                    std::get<1>(mSkelInfos[i]).value[4] = s.second;
            }
        }
    }

    for (auto jn : mSkeleton->getJoints())
        for (int i = 0; i < jn->getNumDofs(); i++)
            jn->setDampingCoefficient(i, mRefSkeleton->getJoint(jn->getName())->getDampingCoefficient(i) * pow(mGlobalRatio, 2.5));

    if (doOptimization == true)
        applySkeletonLength(mSkelInfos, doOptimization);
}

void Character::
    applySkeletonLength(const std::vector<BoneInfo> &info, bool doOptimization)
{
    // const double f0_coeff = 1.5;
    for (auto bone : info)
    {
        std::string name;
        ModifyInfo info;
        std::tie(name, info) = bone;
        modifyLog[mSkeleton->getBodyNode(name)] = info;
    }
    applySkeletonBodyNode(info, mSkeleton);

    Eigen::VectorXd positions = mSkeleton->getPositions();
    mSkeleton->setPositions(Eigen::VectorXd::Zero(mSkeleton->getNumDofs()));
    mSkeleton->computeForwardKinematics(true, false, false);
    mRefSkeleton->setPositions(Eigen::VectorXd::Zero(mSkeleton->getNumDofs()));
    mRefSkeleton->computeForwardKinematics(true, false, false);

    double currentLegLength = mSkeleton->getBodyNode("Pelvis")->getCOM()[1] - mSkeleton->getBodyNode("TalusL")->getCOM()[1];
    double originalLegLength = mRefSkeleton->getBodyNode("Pelvis")->getCOM()[1] - mRefSkeleton->getBodyNode("TalusL")->getCOM()[1];

    if (doOptimization)
    {
        for (int i = 0; i < mMuscles.size(); i++)
        {
            Muscle *mMuscle = mMuscles[i], *mRefMuscle = mRefMuscles[i];
            for (int j = 0; j < mMuscle->GetAnchors().size(); j++)
            {
                Anchor *mAnchor = mMuscle->GetAnchors()[j], *mStdAnchor = mRefMuscle->GetAnchors()[j];
                for (int k = 0; k < mAnchor->bodynodes.size(); k++)
                {
                    BodyNode *mBody = mAnchor->bodynodes[k];
                    int axis = skeletonAxis[mBody->getName()];
                    auto cur = Eigen::Isometry3d(Eigen::Translation3d(mStdAnchor->local_positions[k]));
                    Eigen::Isometry3d tmp = modifyIsometry3d(cur, modifyLog[mBody], axis);
                    mAnchor->local_positions[k] = tmp.translation();
                }
            }
            mMuscle->SetMuscle();
            mMuscle->f0_original = mMuscle->f0 = mRefMuscle->f0 * pow(mMuscle->l_mt0 / mRefMuscle->l_mt0, 1.5);
        }

        {
            // Way Point Optimization
            double eps = 5e-5;
            std::vector<double> derivative;
            if (mLongOpt)
                for (int muscleIdx = 0; muscleIdx < mMuscles.size(); muscleIdx++)
                {
                    auto stdMuscle = mRefMuscles[muscleIdx];
                    auto rtgMuscle = mMuscles[muscleIdx];
                    int numAnchors = rtgMuscle->mAnchors.size();
                    if (muscleToSimpleMotions.find(rtgMuscle->name) == muscleToSimpleMotions.end())
                        continue;
                    const std::vector<SimpleMotion *> &simpleMotions = muscleToSimpleMotions.find(rtgMuscle->name)->second;
                    if (simpleMotions.size() == 0 || numAnchors == 2)
                        continue;

                    std::vector<std::vector<Eigen::Vector3d>> x0(numAnchors - 2);
                    for (int i = 1; i + 1 < numAnchors; i++)
                    {
                        Anchor *anchor = rtgMuscle->mAnchors[i];
                        for (int j = 0; j < anchor->local_positions.size(); j++)
                        {
                            x0[i - 1].push_back(anchor->local_positions[j]);
                        }
                    }

                    // optimize W_r*
                    int rep;
                    for (rep = 0; rep < 20; rep++)
                    {
                        double currentDifference = calculateMetric(stdMuscle, rtgMuscle, simpleMotions, x0);

                        // if waypoint is origin or insertion of any muscle, then it must be fixed.
                        derivative.clear();
                        for (int i = 1; i + 1 < rtgMuscle->mAnchors.size(); i++)
                        {
                            Anchor *anchor = rtgMuscle->mAnchors[i];
                            for (int j = 0; j < anchor->local_positions.size(); j++)
                            {
                                for (int dir = 0; dir < 3; dir++)
                                { // 0:x, 1:y, 2;z
                                    double dx = 0;
                                    anchor->local_positions[j][dir] += eps;
                                    rtgMuscle->SetMuscle();
                                    dx += calculateMetric(stdMuscle, rtgMuscle, simpleMotions, x0);
                                    anchor->local_positions[j][dir] -= eps * 2;
                                    rtgMuscle->SetMuscle();
                                    dx -= calculateMetric(stdMuscle, rtgMuscle, simpleMotions, x0);
                                    anchor->local_positions[j][dir] += eps;
                                    rtgMuscle->SetMuscle();
                                    derivative.push_back(dx / (eps * 2));
                                }
                            }
                        }

                        double alpha = 0.1;
                        // I tried k<100, but result was pretty same as k<16
                        int lineStep;
                        for (lineStep = 0; lineStep < 32; lineStep++)
                        {
                            for (int i = 1, derivativeIdx = 0; i + 1 < rtgMuscle->mAnchors.size(); i++)
                            {
                                Anchor *anchor = rtgMuscle->mAnchors[i];
                                for (int j = 0; j < anchor->local_positions.size(); j++)
                                {
                                    for (int dir = 0; dir < 3; dir++, derivativeIdx++)
                                    { // 0:x, 1:y, 2;z
                                        anchor->local_positions[j][dir] -= alpha * derivative[derivativeIdx];
                                    }
                                }
                            }
                            rtgMuscle->SetMuscle();

                            double nextDifference = calculateMetric(stdMuscle, rtgMuscle, simpleMotions, x0);
                            if (nextDifference < currentDifference * 0.999)
                                break;

                            for (int i = 1, derivativeIdx = 0; i + 1 < rtgMuscle->mAnchors.size(); i++)
                            {
                                Anchor *anchor = rtgMuscle->mAnchors[i];
                                for (int j = 0; j < anchor->local_positions.size(); j++)
                                {
                                    for (int dir = 0; dir < 3; dir++, derivativeIdx++)
                                    { // 0:x, 1:y, 2;z
                                        anchor->local_positions[j][dir] += alpha * derivative[derivativeIdx];
                                    }
                                }
                            }
                            rtgMuscle->SetMuscle();
                            alpha *= 0.5;
                        }
                        if (lineStep == 32)
                            break;
                    }
                    // std::cout << "Muscle " << rtgMuscle->name << " moves " << fRegularizer(rtgMuscle, x0) << " iter " << rep << std::endl;
                }
        }
        for (int i = 0; i < mMuscles.size(); i++)
        {
            Muscle *mMuscle = mMuscles[i], *mRefMuscle = mRefMuscles[i];
            mMuscle->SetMuscle();
            mMuscle->f0_original = mMuscle->f0 = mRefMuscle->f0 * pow(mMuscle->l_mt0 / mRefMuscle->l_mt0, 1.5);
        } //*/
    }

    mSkeleton->setPositions(positions);
    mSkeleton->computeForwardKinematics(true, false, false);
    mRefSkeleton->setPositions(Eigen::VectorXd::Zero(mSkeleton->getNumDofs()));
}

void Character::
    applySkeletonBodyNode(const std::vector<BoneInfo> &info, dart::dynamics::SkeletonPtr skel)
{
    for (auto bone : info)
    {
        std::string name;
        ModifyInfo info;
        std::tie(name, info) = bone;
        int axis = skeletonAxis[name];
        BodyNode *rtgBody = skel->getBodyNode(name);
        BodyNode *stdBody = mRefSkeleton->getBodyNode(name);
        if (rtgBody == NULL)
            continue;

        modifyShapeNode(rtgBody, stdBody, info, axis);

        if (Joint *rtgParent = rtgBody->getParentJoint())
        {
            Joint *stdParent = stdBody->getParentJoint();
            Eigen::Isometry3d up = stdParent->getTransformFromChildBodyNode();
            rtgParent->setTransformFromChildBodyNode(modifyIsometry3d(up, info, axis));
        }

        for (int i = 0; i < rtgBody->getNumChildJoints(); i++)
        {
            Joint *rtgJoint = rtgBody->getChildJoint(i);
            Joint *stdJoint = stdBody->getChildJoint(i);
            Eigen::Isometry3d down = stdJoint->getTransformFromParentBodyNode();
            rtgJoint->setTransformFromParentBodyNode(modifyIsometry3d(down, info, axis, false));
        }
    }
}

double
Character::calculateMetric(Muscle *stdMuscle, Muscle *rtgMuscle, const std::vector<SimpleMotion *> &simpleMotions, const Eigen::EIGEN_VV_VEC3D &x0)
{
    double lambdaShape = 0.1;
    double lambdaLengthCurve = 1.0;
    double lambdaRegularizer = 0.1;

    double shapeTerm = 0;
    double lengthCurveTerm = 0;
    double regularizerTerm = 0;

    double ret = 0.0;
    int numSampling = 50;
    for (SimpleMotion *sm : simpleMotions)
    {
        std::pair<double, double> stdMin, stdMax, rtgMin, rtgMax;
        stdMin = rtgMin = std::pair<double, double>(1e10, 0);
        stdMax = rtgMax = std::pair<double, double>(-1e10, 0);

        for (int rep = 0; rep <= numSampling; rep++)
        {
            double phase = 1.0 * rep / numSampling;
            for (auto [idx, pose] : sm->getPose(phase))
            {
                mRefSkeleton->setPosition(idx, pose);
                mSkeleton->setPosition(idx, pose);
            }

            // shape term
            stdMuscle->Update();
            rtgMuscle->Update();
            shapeTerm += (rep == 0 || rep == numSampling ? 0.5 : 1) * fShape(stdMuscle, rtgMuscle);

            // length curve term
            std::pair<double, double> stdLength = std::make_pair(stdMuscle->length / stdMuscle->l_mt0, phase);
            std::pair<double, double> rtgLength = std::make_pair(rtgMuscle->length / rtgMuscle->l_mt0, phase);
            stdMin = std::min(stdMin, stdLength);
            stdMax = std::max(stdMax, stdLength);
            rtgMin = std::min(rtgMin, rtgLength);
            rtgMax = std::max(rtgMax, rtgLength);
        }
        lengthCurveTerm += fLengthCurve(stdMin.second - rtgMin.second, stdMax.second - rtgMax.second,
                                        (stdMax.first - stdMin.first) - (rtgMax.first - rtgMin.first));

        for (auto [idx, pose] : sm->getPose(0))
        {
            mRefSkeleton->setPosition(idx, 0);
            mSkeleton->setPosition(idx, 0);
        }
    }
    regularizerTerm += fRegularizer(rtgMuscle, x0);

    int dof = mRefSkeleton->getNumDofs();
    mRefSkeleton->setPositions(Eigen::VectorXd::Zero(dof));
    mSkeleton->setPositions(Eigen::VectorXd::Zero(dof));

    return lambdaShape * shapeTerm / numSampling / simpleMotions.size() + lambdaLengthCurve * lengthCurveTerm / simpleMotions.size() + lambdaRegularizer * regularizerTerm;
}

double
Character::fShape(Muscle *stdMuscle, Muscle *rtgMuscle)
{
    double ret = 0;
    int cnt = 0;
    for (int i = 1; i + 1 < stdMuscle->mAnchors.size(); i++)
    {
        auto std_bn = stdMuscle->mAnchors[i]->bodynodes[0];
        auto rtg_bn = rtgMuscle->mAnchors[i]->bodynodes[0];
        Eigen::Matrix3d std_bn_inv = std_bn->getTransform().linear().transpose();
        Eigen::Matrix3d rtg_bn_inv = rtg_bn->getTransform().linear().transpose();
        // Eigen::Isometry3d std_bn_inv = std_bn->getTransform().inverse();
        // Eigen::Isometry3d rtg_bn_inv = rtg_bn->getTransform().inverse();

        Eigen::Vector3d stdVector, rtgVector;
        stdVector = std_bn_inv * (stdMuscle->mCachedAnchorPositions[i + 1] - stdMuscle->mCachedAnchorPositions[i]);
        rtgVector = rtg_bn_inv * (rtgMuscle->mCachedAnchorPositions[i + 1] - rtgMuscle->mCachedAnchorPositions[i]);
        stdVector.normalize();
        rtgVector.normalize();
        ret += 0.5 * (stdVector.cross(rtgVector)).norm();

        stdVector = std_bn_inv * (stdMuscle->mCachedAnchorPositions[i - 1] - stdMuscle->mCachedAnchorPositions[i]);
        rtgVector = rtg_bn_inv * (rtgMuscle->mCachedAnchorPositions[i - 1] - rtgMuscle->mCachedAnchorPositions[i]);
        stdVector.normalize();
        rtgVector.normalize();
        ret += 0.5 * (stdVector.cross(rtgVector)).norm();

        cnt += 1;
    }
    return cnt ? ret / cnt : 0;
}

double
Character::
    getSkelParamValue(std::string skel_name)
{

    if (skel_name == "global")
        return mGlobalRatio;
    else
    {
        for (auto s_i : mSkelInfos)
            if (std::get<0>(s_i) == skel_name)
            {
                if (skel_name.find("Arm") != std::string::npos)
                    return std::get<1>(s_i).value[0] / mGlobalRatio;
                else
                    return std::get<1>(s_i).value[1] / mGlobalRatio;
            }
    }
    std::cout << "[DEBUG] Error" << std::endl;
    exit(-1);
    return -1;
}

double
Character::
    getTorsionValue(std::string skel_name)
{
    for (auto s_i : mSkelInfos)
        if (std::get<0>(s_i) == skel_name)
            return std::get<1>(s_i).value[4];
    return 0.0;
}

Eigen::VectorXd Character::
    posToSixDof(Eigen::VectorXd pos)
{
    Eigen::VectorXd displacement;
    int p_size = 0;
    for (auto jn : mSkeleton->getJoints())
    {
        if (jn->getNumDofs() == 3 || jn->getNumDofs() == 6)
            p_size += 3;
        p_size += jn->getNumDofs();
    }
    displacement.resize(p_size);
    int p_idx = 0;
    for (auto jn : mSkeleton->getJoints())
    {
        if (jn->getNumDofs() == 0)
            continue;

        if (jn->getNumDofs() == 1)
            displacement[p_idx++] = pos[jn->getIndexInSkeleton(0)];
        // jn->getPosition(0);
        else if (jn->getNumDofs() == 3)
        {
            Eigen::Matrix3d r = BallJoint::convertToRotation(pos.segment(jn->getIndexInSkeleton(0), jn->getNumDofs()));
            displacement.segment<6>(p_idx) << r(0, 0), r(0, 1), r(0, 2), r(1, 0), r(1, 1), r(1, 2);
            p_idx += 6;
        }
        else if (jn->getNumDofs() == 6)
        {
            Eigen::Isometry3d i = FreeJoint::convertToTransform(pos.segment(jn->getIndexInSkeleton(0), jn->getNumDofs()));
            Eigen::Matrix3d r = i.linear();
            Eigen::Vector3d t = i.translation();
            displacement.segment<6>(p_idx) << r(0, 0), r(0, 1), r(0, 2), r(1, 0), r(1, 1), r(1, 2);
            p_idx += 6;
            displacement.segment<3>(p_idx) = t;
            p_idx += 3;
        }
    }
    return displacement;
}

Eigen::VectorXd Character::sixDofToPos(Eigen::VectorXd raw_pos)
{
    int p_idx = 0;
    Eigen::VectorXd pos = mSkeleton->getPositions();

    for (auto jn : mSkeleton->getJoints())
    {
        if (jn->getNumDofs() == 0)
            continue;

        int idx = jn->getIndexInSkeleton(0);
        if (jn->getNumDofs() == 1)
            pos[idx] = raw_pos[p_idx++];
        else if (jn->getNumDofs() == 3)
        {
            // convert 6d-rotation to angle axis rotation 
            Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
            Eigen::Vector3d v1 = raw_pos.segment<3>(p_idx).normalized();
            p_idx += 3;
            Eigen::Vector3d v2 = raw_pos.segment<3>(p_idx);
            p_idx += 3;
            Eigen::Vector3d v3 = v1.cross(v2).normalized();
            v2 = v3.cross(v1).normalized();

            r.row(0) = v1;
            r.row(1) = v2;
            r.row(2) = v3;
            pos.segment<3>(idx) = BallJoint::convertToPositions(r);



        }
        else if (jn->getNumDofs() == 6)
        {
            Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
            Eigen::Vector3d v1 = raw_pos.segment<3>(p_idx).normalized();
            p_idx += 3;
            Eigen::Vector3d v2 = raw_pos.segment<3>(p_idx);
            p_idx += 3;
            Eigen::Vector3d v3 = v1.cross(v2).normalized();
            v2 = v3.cross(v1).normalized();

            r.row(0) = v1;
            r.row(1) = v2;
            r.row(2) = v3;
            pos.segment<3>(idx) = BallJoint::convertToPositions(r);
            idx += 3;
            pos.segment<3>(idx) = raw_pos.segment<3>(p_idx);
            p_idx += 3;
        }
    }

    return pos;
}
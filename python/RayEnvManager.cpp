#include "Environment.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>

namespace py = pybind11;

py::array_t<float> toNumPyArray(const Eigen::VectorXf &vec)
{
    int n = vec.rows();

    py::array_t<float> array(n);
    py::buffer_info buf = array.request(true);
    float *ptr = reinterpret_cast<float *>(buf.ptr);

    for (int i = 0; i < n; i++)
        ptr[i] = (float)vec(i);

    return array;
}

py::array_t<float> toNumPyArray(const Eigen::VectorXd &vec)
{
    int n = vec.rows();

    py::array_t<float> array(n);
    py::buffer_info buf = array.request(true);
    float *ptr = reinterpret_cast<float *>(buf.ptr);

    for (int i = 0; i < n; i++)
        ptr[i] = (float)vec(i);

    return array;
}

py::array_t<float> toNumPyArray(const Eigen::MatrixXd &matrix)
{
    int n = matrix.rows();
    int m = matrix.cols();

    py::array_t<float> array({n, m});
    py::buffer_info buf = array.request(true);
    float *ptr = reinterpret_cast<float *>(buf.ptr);

    int index = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            ptr[i * m + j] = (float)matrix(i, j);
        }
    }

    return array;
}

Eigen::VectorXd toEigenVector(const py::array_t<float> &array)
{
    Eigen::VectorXd vec(array.shape(0));

    py::buffer_info buf = array.request();
    float *srcs = reinterpret_cast<float *>(buf.ptr);

    for (int i = 0; i < array.shape(0); i++)
        vec(i) = (double)srcs[i];

    return vec;
}

class RayEnvManager : public Environment
{
public:
    RayEnvManager(std::string metadata) : Environment()
    {
        Environment::initialize(metadata);
    }
    py::array_t<float> getState() { return toNumPyArray(Environment::getState()); }
    py::array_t<float> getAction() { return toNumPyArray(Environment::getAction()); }
    py::list getRandomMuscleTuple()
    {
        MuscleTuple mt = Environment::getRandomMuscleTuple();
        Eigen::VectorXd dt = Environment::getRandomDesiredTorque();

        py::list py_mt;
        py_mt.append(toNumPyArray(dt));
        py_mt.append(toNumPyArray(mt.JtA_reduced));
        py_mt.append(toNumPyArray(mt.JtA));
        if (Environment::getUseCascading())
        {
            py_mt.append(toNumPyArray(Environment::getRandomPrevOut()));
            py_mt.append(toNumPyArray(Environment::getRandomWeight()));
        }
        return py_mt;
    }
    // void setAction(py::object action) { Environment::setAction(action.cast<Eigen::VectorXd>()); }
    void setAction(py::array_t<float> action) { Environment::setAction(toEigenVector(action)); }
    void step() { Environment::step(Environment::getSimulationHz() / Environment::getControlHz()); }
    int getNumMuscles() { return Environment::getCharacter(0)->getMuscles().size(); }

    int getNumMuscleDof() { return Environment::getCharacter(0)->getNumMuscleRelatedDof(); }

    py::array_t<float> getParamState() { return toNumPyArray(Environment::getParamState()); }
    py::array_t<float> getNormalizedParamState() { return toNumPyArray(Environment::getNormalizedParamState(Environment::getParamMin(), Environment::getParamMax())); }
    py::array_t<float> getPositions() { return toNumPyArray(Environment::getCharacter(0)->getSkeleton()->getPositions()); }
    py::array_t<float> posToSixDof(py::array_t<float> pos) { return toNumPyArray(Environment::getCharacter(0)->posToSixDof(toEigenVector(pos))); }
    py::array_t<float> sixDofToPos(py::array_t<float> raw_pos) { return toNumPyArray(Environment::getCharacter(0)->sixDofToPos(toEigenVector(raw_pos))); }

    py::array_t<float> getMirrorParamState(py::array_t<float> param)
    {

        Eigen::VectorXd cur_paramstate = Environment::getParamState();
        Environment::setNormalizedParamState(toEigenVector(param), false, true);
        Eigen::VectorXd res = Environment::getNormalizedParamState(Environment::getParamMin(), Environment::getParamMax(), true);
        Environment::setParamState(cur_paramstate, false, true);
        return toNumPyArray(res);
    }

    py::array_t<float> getMirrorPositions(py::array_t<float> pos) { return toNumPyArray(getCharacter(0)->getMirrorPosition(toEigenVector(pos))); }

    py::array_t<float> getParamStateFromNormalized(py::array_t<float> normalized_param)
    {
        return toNumPyArray(Environment::getParamStateFromNormalized(toEigenVector(normalized_param)));
    }

    py::array_t<float> getNormalizedParamStateFromParam(py::array_t<float> param)
    {
        return toNumPyArray(Environment::getNormalizedParamStateFromParam(toEigenVector(param)));
    }

    // getParamSample
    py::array_t<float> getNormalizedParamSample()
    {
        return toNumPyArray(Environment::getNormalizedParamStateFromParam(Environment::getParamSample()));
    }
};

PYBIND11_MODULE(pysim, m)
{
    // py::class_<dart::dynamics::Skeleton, dart::dynamics::SkeletonPtr>(m, "Skeleton");
    py::class_<RayEnvManager>(m, "RayEnvManager")
        .def(py::init<std::string>())
        // .def("initialize", &RayEnvManager::initialize)
        .def("setAction", &RayEnvManager::setAction)
        .def("step", &RayEnvManager::step)
        .def("reset", &RayEnvManager::reset)
        .def("isEOE", &RayEnvManager::isEOE)
        .def("getReward", &RayEnvManager::getReward)
        .def("getState", &RayEnvManager::getState)
        .def("getAction", &RayEnvManager::getAction)

        .def("getNumAction", &RayEnvManager::getNumAction)
        .def("getNumActuatorAction", &RayEnvManager::getNumActuatorAction)
        .def("getNumMuscles", &RayEnvManager::getNumMuscles)
        .def("getNumMuscleDof", &RayEnvManager::getNumMuscleDof)

        .def("getMetadata", &RayEnvManager::getMetadata)
        .def("getRandomMuscleTuple", &RayEnvManager::getRandomMuscleTuple)
        .def("getUseMuscle", &RayEnvManager::getUseMuscle)
        .def("setMuscleNetwork", &RayEnvManager::setMuscleNetwork)
        .def("setMuscleNetworkWeight", &RayEnvManager::setMuscleNetworkWeight)
        .def("isTwoLevelController", &RayEnvManager::isTwoLevelController)

        .def("getLearningStd", &RayEnvManager::getLearningStd)
        .def("getUseCascading", &RayEnvManager::getUseCascading)

        .def("getParamState", &RayEnvManager::getParamState)
        .def("updateParamState", &RayEnvManager::updateParamState)
        // For Rollout (Forward GaitNet)
        .def("getNormalizedParamState", &RayEnvManager::getNormalizedParamState)
        .def("getNormalizedPhase", &RayEnvManager::getNormalizedPhase)
        .def("getWorldPhase", &RayEnvManager::getWorldPhase)

        .def("getPositions", &RayEnvManager::getPositions)

        .def("posToSixDof", &RayEnvManager::posToSixDof)
        .def("sixDofToPos", &RayEnvManager::sixDofToPos)

        .def("getMirrorParamState", &RayEnvManager::getMirrorParamState)
        .def("getMirrorPositions", &RayEnvManager::getMirrorPositions)
        .def("getParamStateFromNormalized", &RayEnvManager::getParamStateFromNormalized)
        .def("getNormalizedParamStateFromParam", &RayEnvManager::getNormalizedParamStateFromParam)
        .def("getNumKnownParam", &RayEnvManager::getNumKnownParam)

        .def("getNormalizedParamSample", &RayEnvManager::getNormalizedParamSample)

        // get min v
        .def("getParamMin", &RayEnvManager::getParamMin)
        .def("getParamMax", &RayEnvManager::getParamMax)

        ;
}

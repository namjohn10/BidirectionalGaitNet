#include "GLFWApp.h"
#include <pybind11/embed.h>

namespace py = pybind11;

int main(int argc, char **argv)
{
    pybind11::scoped_interpreter guard{};
    Environment *env = new Environment();
    GLFWApp app(argc, argv);
    app.setEnv(env);
    app.startLoop();

    return -1;
}
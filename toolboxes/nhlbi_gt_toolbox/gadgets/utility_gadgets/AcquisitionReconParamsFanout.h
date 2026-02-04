#pragma once
#include <Node.h>

#include <ismrmrd/ismrmrd.h>

#include <Fanout.h>
#include <Types.h>
#include "reconParams.h"
using namespace Gadgetron;
using namespace Gadgetron::Core;

using AcquisitionReconParamsFanout = Gadgetron::Core::Parallel::Fanout<variant<Acquisition, Waveform,Gadgetron::reconParams,
                                         std::vector<std::vector<std::vector<std::vector<size_t>>>>>>;

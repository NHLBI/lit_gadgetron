#pragma once
#include <Node.h>

#include <ismrmrd/ismrmrd.h>

#include <Fanout.h>
#include <Types.h>

using namespace Gadgetron;
using namespace Gadgetron::Core;

using AcquisitionWaveformFanout = Gadgetron::Core::Parallel::Fanout<variant<Acquisition, Waveform>>;

#pragma once
#include <Node.h>

#include <ismrmrd/ismrmrd.h>

#include <Fanout.h>
#include <Types.h>
#include "../spiral/SpiralBuffer.h"
#include <cuNDArray.h>

using namespace Gadgetron;
using namespace Gadgetron::Core;

using SpiralBufferFanout = Gadgetron::Core::Parallel::Fanout<Gadgetron::SpiralBuffer<cuNDArray, float_complext, 2>>;

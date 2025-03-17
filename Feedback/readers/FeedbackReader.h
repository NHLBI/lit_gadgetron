#pragma once
#include "../MessageID.h"
#include "Reader.h"
namespace Gadgetron::Core::Readers {

    class FeedbackReader : public Gadgetron::Core::Reader {
    public:
        virtual Message read(std::istream &stream) override;
        virtual uint16_t slot() override;
    };
};


#include "HeaderConnection.h"

#include <map>
#include <iostream>

#include "storage.h"
#include "Handlers.h"
#include "StreamConnection.h"
#include "VoidConnection.h"
#include "config/Config.h"

#include "io/primitives.h"
#include "Context.h"
#include "MessageID.h"
//PD
#include <filesystem>
#include <ismrmrd/ismrmrd.h>
#include <ismrmrd/dataset.h>
#include <ismrmrd/xml.h>
#include <ismrmrd/waveform.h>
//PD
#define CONFIG_ERROR "Received second config file. Only one allowed."

namespace {

    using namespace Gadgetron::Core;
    using namespace Gadgetron::Core::IO;
    using namespace Gadgetron::Server::Connection;
    using namespace Gadgetron::Server::Connection::Handlers;

    using Header = Gadgetron::Core::StreamContext::Header;

    class HeaderHandler : public Handler {
    public:
        explicit HeaderHandler(
                std::function<void(Header)> header_callback
        ) : header_callback(std::move(header_callback)) {}

        void handle(std::istream &stream, OutputChannel&) override {
            std::string raw_header(read_string_from_stream<uint32_t>(stream));

            ISMRMRD::IsmrmrdHeader header{};
            
            //GDEBUG_STREAM("HEADER" << raw_header.c_str());
            ISMRMRD::deserialize(raw_header.c_str(), header);
            //PD
            /* tSequenceVariant in the header give information about the hash of the pulseq seq file
            Loop through all seq files in a directory and find the correct seq file (hash is in the last line )
            */
            if (header.userParameters) {
                ISMRMRD::UserParameters user_params = header.userParameters.get();
                std::vector<ISMRMRD::UserParameterString> strings = user_params.userParameterString;
                std::vector<ISMRMRD::UserParameterString>::iterator it;

                for (it = strings.begin(); it != strings.end(); ++it) {
                    if (it->name == "tSequenceVariant") {
                        GDEBUG("Hash bstar found %s found\n", it->value.c_str());
                        std::string hash_bstar=it->value.c_str();
                        std::string bstar_folder_traj="/opt/data/bstar_traj/";
                        // Loop through all .seq files in bstar_folder_traj
                        for (const auto& entry : std::filesystem::directory_iterator(bstar_folder_traj)) {
                            
                            if (entry.path().extension()==".seq"){
                                //Find the hash (Last line of the .seq file)
                                std::ifstream fin;
                                fin.open(entry.path().string());
                                if(fin.is_open()) {
                                    fin.seekg(-2,std::ios_base::end);                // go to one spot before the EOF

                                    bool keepLooping = true;
                                    while(keepLooping) {
                                        char ch;
                                        fin.get(ch);                            // Get current byte's data

                                        if((int)fin.tellg() <= 1) {             // If the data was at or before the 0th byte
                                            fin.seekg(0);                       // The first line is the last line
                                            keepLooping = false;                // So stop there
                                        }
                                        else if(ch == '\n') {                   // If the data was a newline
                                            keepLooping = false;                // Stop at the current position.
                                        }
                                        else {                                  // If the data was neither a newline nor at the 0 byte
                                            fin.seekg(-2,std::ios_base::cur);        // Move to the front of that data, then to the front of the data before it
                                        }
                                    }

                                    std::string lastLine;            
                                    std::getline(fin,lastLine);                      // Read the current line
                                    fin.close();

                                    // Comparison hash of the data with hash of the .seq file
                                    if(lastLine.find(hash_bstar) != std::string::npos){
                                        GDEBUG_STREAM("Seq file found with hash"<< lastLine);
                                        
                                        std::string traj_h5=(bstar_folder_traj +std::string("traj_") + entry.path().stem().string() + std::string(".h5"));
                                        GDEBUG_STREAM("Pulseq trajectory filepath " << traj_h5);
                                        
                                        std::string xml_config;
                                        std::string hdf5_in_group="/dataset";
                                        std::shared_ptr<ISMRMRD::Dataset> ismrmrd_dataset= std::shared_ptr<ISMRMRD::Dataset>(new ISMRMRD::Dataset(traj_h5.c_str(), hdf5_in_group.c_str(), false));
                                        ismrmrd_dataset->readHeader(xml_config);
                                        ISMRMRD::IsmrmrdHeader h_traj;
                                        
                                        ISMRMRD::deserialize(xml_config.c_str(), h_traj);
                                        // MODIFying Header matrixSize and FOV : Only reconSpace.matrixSize/FOV are correct in the trajectory file
                                        auto factor_0r = float((size_t(round(float(h_traj.encoding.front().reconSpace.matrixSize.x) / 32.0))) * 32.0) / float(size_t(h_traj.encoding.front().reconSpace.matrixSize.x));
                                        auto factor_1r = float((size_t(round(float(h_traj.encoding.front().reconSpace.matrixSize.y) / 32.0))) * 32.0) / float(size_t(h_traj.encoding.front().reconSpace.matrixSize.y));
                                        auto factor_2r = float((size_t(round(float(h_traj.encoding.front().reconSpace.matrixSize.z) / 32.0))) * 32.0) / float(size_t(h_traj.encoding.front().reconSpace.matrixSize.z));
                                        GDEBUG_STREAM("Recon Matrix traj : X " << h_traj.encoding.front().reconSpace.matrixSize.x << " Y " << h_traj.encoding.front().reconSpace.matrixSize.y << " Z " << h_traj.encoding.front().reconSpace.matrixSize.z);
                                        GDEBUG_STREAM("Factor traj : X " << factor_0r << " Y " << factor_1r << " Z " << factor_2r);
                                        
                                        factor_0r = factor_0r == 0 ? 1 : factor_0r;
                                        factor_1r = factor_1r == 0 ? 1 : factor_1r;
                                        factor_2r = factor_2r == 0 ? 1 : factor_2r;
                                        GDEBUG_STREAM("Raw Header information :");
                                        GDEBUG_STREAM("Encoded Matrix: X " << header.encoding.front().encodedSpace.matrixSize.x << " Y " << header.encoding.front().encodedSpace.matrixSize.y << " Z " << header.encoding.front().encodedSpace.matrixSize.z);
                                        GDEBUG_STREAM("Recon Matrix: X " << header.encoding.front().reconSpace.matrixSize.x << " Y " << header.encoding.front().reconSpace.matrixSize.y << " Z " << header.encoding.front().reconSpace.matrixSize.z);
                                        GDEBUG_STREAM("Encoded FOV: X " << header.encoding.front().encodedSpace.fieldOfView_mm.x << " Y " << header.encoding.front().encodedSpace.fieldOfView_mm.y << " Z " << header.encoding.front().encodedSpace.fieldOfView_mm.z);
                                        GDEBUG_STREAM("Recon FOV: X " << header.encoding.front().reconSpace.fieldOfView_mm.x << " Y " << header.encoding.front().reconSpace.fieldOfView_mm.y << " Z " << header.encoding.front().reconSpace.fieldOfView_mm.z);
                                        GDEBUG_STREAM("Encoding Limits: Encoded step 1 max " << header.encoding.at(0).encodingLimits.kspace_encoding_step_1.get().maximum  << " Encoded step 2 max " << header.encoding.at(0).encodingLimits.kspace_encoding_step_2.get().maximum  );

                                        header.encoding.front().encodedSpace.matrixSize.x=size_t(h_traj.encoding.front().reconSpace.matrixSize.x* factor_0r);
                                        header.encoding.front().encodedSpace.matrixSize.y=size_t(h_traj.encoding.front().reconSpace.matrixSize.y* factor_1r);
                                        header.encoding.front().encodedSpace.matrixSize.z=size_t(h_traj.encoding.front().reconSpace.matrixSize.z* factor_2r);

                                        header.encoding.front().reconSpace.matrixSize.x=size_t(h_traj.encoding.front().reconSpace.matrixSize.x* factor_0r);
                                        header.encoding.front().reconSpace.matrixSize.y=size_t(h_traj.encoding.front().reconSpace.matrixSize.y* factor_1r);
                                        header.encoding.front().reconSpace.matrixSize.z=size_t(h_traj.encoding.front().reconSpace.matrixSize.z* factor_2r);

                                        header.encoding.front().encodedSpace.fieldOfView_mm.x=h_traj.encoding.front().reconSpace.fieldOfView_mm.x;
                                        header.encoding.front().encodedSpace.fieldOfView_mm.y=h_traj.encoding.front().reconSpace.fieldOfView_mm.y;
                                        header.encoding.front().encodedSpace.fieldOfView_mm.z=h_traj.encoding.front().reconSpace.fieldOfView_mm.z;

                                        header.encoding.front().reconSpace.fieldOfView_mm.x=h_traj.encoding.front().reconSpace.fieldOfView_mm.x;
                                        header.encoding.front().reconSpace.fieldOfView_mm.y=h_traj.encoding.front().reconSpace.fieldOfView_mm.y;
                                        header.encoding.front().reconSpace.fieldOfView_mm.z=h_traj.encoding.front().reconSpace.fieldOfView_mm.z;


                                        // Modifying the header encodingLimits :
                                        
                                        header.encoding.at(0).encodingLimits.kspace_encoding_step_1.get().maximum =h_traj.encoding.front().encodingLimits.kspace_encoding_step_1.get().maximum;
                                        header.encoding.at(0).encodingLimits.kspace_encoding_step_2.get().maximum =h_traj.encoding.front().encodingLimits.segment.get().maximum;

                                        GDEBUG_STREAM("Header modified information :");
                                        GDEBUG_STREAM("Encoded Matrix: X " << header.encoding.front().encodedSpace.matrixSize.x << " Y " << header.encoding.front().encodedSpace.matrixSize.y << " Z " << header.encoding.front().encodedSpace.matrixSize.z);
                                        GDEBUG_STREAM("Recon Matrix: X " << header.encoding.front().reconSpace.matrixSize.x << " Y " << header.encoding.front().reconSpace.matrixSize.y << " Z " << header.encoding.front().reconSpace.matrixSize.z);
                                        GDEBUG_STREAM("Encoded FOV: X " << header.encoding.front().encodedSpace.fieldOfView_mm.x << " Y " << header.encoding.front().encodedSpace.fieldOfView_mm.y << " Z " << header.encoding.front().encodedSpace.fieldOfView_mm.z);
                                        GDEBUG_STREAM("Recon FOV: X " << header.encoding.front().reconSpace.fieldOfView_mm.x << " Y " << header.encoding.front().reconSpace.fieldOfView_mm.y << " Z " << header.encoding.front().reconSpace.fieldOfView_mm.z);
                                        GDEBUG_STREAM("Encoding Limits: Encoded step 1 max " << header.encoding.at(0).encodingLimits.kspace_encoding_step_1.get().maximum  << " Encoded step 2 max " << header.encoding.at(0).encodingLimits.kspace_encoding_step_2.get().maximum  );

                                        break;
                                    }
                                }
                                
                            }

                        }
                    }
                }
            }
            //PD
            header_callback(header);
        }

    private:
        std::function<void(Header)> header_callback;
    };

    class HeaderContext {
    public:
        Gadgetron::Core::optional<Header> header;
        const StreamContext::Paths paths;
    };

    std::map<uint16_t, std::unique_ptr<Handler>> prepare_handlers(
            std::function<void()> close,
            HeaderContext &context
    ) {
        std::map<uint16_t, std::unique_ptr<Handler>> handlers{};

        auto header_callback = [=, &context](Header header) {
            context.header = header;
            close();
        };

        handlers[FILENAME] = std::make_unique<ErrorProducingHandler>(CONFIG_ERROR);
        handlers[CONFIG]   = std::make_unique<ErrorProducingHandler>(CONFIG_ERROR);
        handlers[HEADER]   = std::make_unique<HeaderHandler>(header_callback);
        handlers[TEXT]     = std::make_unique<TextLoggerHandler>();
        handlers[QUERY]    = std::make_unique<QueryHandler>();
        handlers[CLOSE]    = std::make_unique<CloseHandler>(close);

        return handlers;
    }
}

namespace Gadgetron::Server::Connection::HeaderConnection {

    void process(
            std::iostream &stream,
            const Core::StreamContext::Paths &paths,
            const Core::StreamContext::Args &args,
            const Core::StreamContext::StorageAddress& storage_address,
            const Config &config,
            ErrorHandler &error_handler
    ) {
        GINFO_STREAM("Connection state: [HEADER]");

        HeaderContext context{
                Core::none,
                paths
        };

        auto channel = make_channel<MessageChannel>();

        std::thread input_thread = start_input_thread(
                stream,
                std::move(channel.output),
                [&](auto close) { return prepare_handlers(close, context); },
                error_handler
        );

        std::thread output_thread = start_output_thread(
                stream,
                std::move(channel.input),
                default_writers,
                error_handler
        );

        input_thread.join();
        output_thread.join();

        auto header = context.header.value_or(Header());
        StreamContext stream_context{
            header,
            paths,
            args,
            storage_address,
            setup_storage_spaces(storage_address, header)
        };

        auto process = context.header ? StreamConnection::process : VoidConnection::process;
        process(stream, stream_context, config, error_handler);
    }
}
#pragma once

#include <gvirtus/common/LD_Lib.h>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <stdlib.h> /* getenv */
#include "Communicator.h"
#include "Endpoint.h"
#include "Endpoint_Tcp.h"

namespace gvirtus::communicators {
    class CommunicatorFactory {
    public:
        static std::shared_ptr<common::LD_Lib<Communicator, std::shared_ptr<Endpoint>>>
        get_communicator(std::shared_ptr<Endpoint> end, bool secure = false) {
            std::shared_ptr<common::LD_Lib<Communicator, std::shared_ptr<Endpoint>>> dl;
            std::string gvirtus_home = CommunicatorFactory::getGVirtuSHome();

            // Supported unsecure communicators
            std::vector<std::string> unsecureMatches = {"tcp", "http", "oldtcp", "ws"};
            // Supported secure communicators
            std::vector<std::string> secureMatches = {"https", "wss"};

            // Is the desired communicator a secure communicator?
            if (not secure) {
                // No, then search it into the supported unsecure communicators vector
                auto foundIt = std::find(unsecureMatches.begin(), unsecureMatches.end(), end->protocol());
                // Is the desired communicator supported?
                if (foundIt == unsecureMatches.end()) {
                    // No, then throw an exception.
                    throw std::runtime_error("Unsecure communicator not supported");
                }
            }
            else {
                // Yes, then search it into the supported secure communicators vector
                auto foundIt = std::find(secureMatches.begin(), secureMatches.end(), end->protocol());
                // Is the desired communicator supported?
                if (foundIt == secureMatches.end()) {
                    // No, then throw an exception.
                    throw std::runtime_error("Secure communicator not supported");
                }
            }

            dl = std::make_shared<common::LD_Lib<Communicator, std::shared_ptr<Endpoint>>>(
                    gvirtus_home + "/lib/libgvirtus-communicators-" + end->protocol() + ".so",
                    "create_communicator");

            dl->build_obj(end);
            return dl;
        }

    private:
        static std::string getEnvVar(std::string const &key) {
            char *val = getenv(key.c_str());
            return val == NULL ? std::string("") : std::string(val);
        }

        static std::string getGVirtuSHome() {
            std::string gvirtus_home = CommunicatorFactory::getEnvVar("GVIRTUS_HOME");
            return gvirtus_home;
        }

        static std::string getConfigFile() {
            // Get the GVIRTUS_CONFIG environment varibale
            std::string config_path = CommunicatorFactory::getEnvVar("GVIRTUS_CONFIG");

            // Check if the configuration file is defined
            if (config_path == "") {
                // Check if the configuration file is in the GVIRTUS_HOME directory
                config_path = CommunicatorFactory::getEnvVar("GVIRTUS_HOME") + "/etc/properties.json";

                if (config_path == "") {
                    // Finally consider the current directory
                    config_path = "./properties.json";
                }
            }
            return config_path;
        }
    };
}  // namespace gvirtus::communicators

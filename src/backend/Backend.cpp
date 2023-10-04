#include "gvirtus/backend/Backend.h"

#include <gvirtus/communicators/CommunicatorFactory.h>
#include <gvirtus/communicators/EndpointFactory.h>

#include <sys/wait.h>
#include <unistd.h>

using gvirtus::backend::Backend;

Backend::Backend(const fs::path &path) {
    logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("Backend"));

    // Set the logging level
    log4cplus::LogLevel logLevel = log4cplus::INFO_LOG_LEVEL;

    char *val = getenv("GVIRTUS_LOGLEVEL");
    std::string logLevelString = (val == NULL ? std::string("") : std::string(val));

    if (logLevelString != "") {
        logLevel = std::stoi(logLevelString);
    }

    logger.setLogLevel(logLevel);

    // json
    if (fs::exists(path) && fs::is_regular_file(path) &&
        path.extension() == ".json") {
        LOG4CPLUS_DEBUG(logger, "✓ - " << fs::path(__FILE__).filename() << ":" << __LINE__ << ":"
                                       << " Json file has been loaded.");

        _properties = common::JSON<Property>(path).parser();
        _children.reserve(_properties.endpoints());

        if (_properties.endpoints() > 1) {
            LOG4CPLUS_INFO(logger, "🛈  - Application serves on "
                    << _properties.endpoints()
                    << " several endpoint");
        }

        for (int i = 0; i < _properties.endpoints(); i++) {
            _children.push_back(std::make_unique<Process>(
                    communicators::CommunicatorFactory::get_communicator(
                            communicators::EndpointFactory::get_endpoint(path),
                            _properties.secure()),
                    _properties.plugins().at(i)));
        }

    } else {
        LOG4CPLUS_ERROR(logger, "✖ - " << fs::path(__FILE__).filename() << ":"
                                       << __LINE__ << ":"
                                       << " json path error: no such file.");
        exit(EXIT_FAILURE);
    }
}

void Backend::Start() {
    // std::function<void(std::unique_ptr<gvirtus::Thread> & children)> task =
    // [this](std::unique_ptr<gvirtus::Thread> &children) {
    //   LOG4CPLUS_DEBUG(logger, "✓ - [Thread " << std::this_thread::get_id() <<
    //   "]: Started."); children->Start(); LOG4CPLUS_DEBUG(logger, "✓ - [Thread "
    //   << std::this_thread::get_id() << "]: Finished.");
    // };
    LOG4CPLUS_DEBUG(logger, "✓ - Backend::Start() called.");

    int pid = 0;
    // _children definition: "std::vector<std::unique_ptr<Process>> _children"
    for (int i = 0; i < _children.size(); i++) {
        if ((pid = fork()) == 0) {
            _children[i]->Start();
            break;
        }
    }

    pid_t pid_wait = 0;
    int stat_loc;
    if (pid != 0) {
        signal(SIGINT, SIG_IGN);
        signal(SIGHUP, SIG_IGN);

        while ((pid_wait = wait(&stat_loc)) > 0) {
            std::stringstream debug_msg;
            debug_msg << "[Process " << getpid() << "] ";

            if (WIFEXITED(stat_loc))
                debug_msg << "terminated normally (WIFEXITED TRUE)";
            if (WIFSIGNALED(stat_loc))
                debug_msg << "terminated due to receipt of a signal (WIFSIGNALED TRUE)";
            if (WIFSTOPPED(stat_loc))
                debug_msg << "has stopped and can be restarted (WIFSTOPPED TRUE)";

            LOG4CPLUS_DEBUG(logger, "✓ - " << __LINE__ << ": " << debug_msg.str());
        }
    }

    LOG4CPLUS_DEBUG(logger, "✓ - Backend::Start() returned.");
}

void Backend::EventOccurred(std::string &event, void *object) {
    LOG4CPLUS_DEBUG(logger, "✓ - EventOccurred: " << event);
}

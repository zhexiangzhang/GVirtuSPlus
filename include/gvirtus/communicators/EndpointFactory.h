#pragma once

#include <gvirtus/common/JSON.h>
#include <memory>
#include <nlohmann/json.hpp>
#include "Endpoint.h"
#include "Endpoint_Tcp.h"

namespace gvirtus::communicators {
class EndpointFactory {
 public:
  static std::shared_ptr<Endpoint> get_endpoint(const fs::path &json_path) {
      nlohmann::json j = {
              {"communicator", {
                                       {
                                               {"endpoint", {
                                                                    {"suite", "tcp/ip"},
                                                                    {"protocol", "tcp"},
                                                                    {"server_address", "127.0.0.1"},
                                                                    {"port", "9999"}
                                                            }},
                                               {"plugins", {"cudart", "cublas", "curand", "cudnn"}}
                                       }
                               }},
              {"secure_application", false}
      };

      Endpoint_Tcp end;
      from_json(j, end);  // 使用 from_json 函数填充 Endpoint_Tcp 对象

      // 包装 Endpoint_Tcp 对象到 shared_ptr
      std::shared_ptr<Endpoint> ptr = std::make_shared<Endpoint_Tcp>(end);

      ind_endpoint++;  // 更新索引，虽然在这个场景中可能不需要

      return ptr;
  }

  static int index() { return ind_endpoint; }

 private:
  static int ind_endpoint;
};
}  // namespace gvirtus::communicators

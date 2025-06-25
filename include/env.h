//
// Created by richard on 6/25/25.
//

#ifndef ENV_H
#define ENV_H

#include <string>
#include <optional>

namespace atp {

void set_env(const char* name, const char* value, bool overwrite);

std::optional<std::string> get_env(const char* name) noexcept;

bool has_env(const char* name) noexcept;

std::optional<bool> check_env(const char* name);

}

#endif //ENV_H

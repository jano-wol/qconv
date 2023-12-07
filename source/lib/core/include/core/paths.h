#ifndef BARBAKAN_CORE_PATHS_H_
#define BARBAKAN_CORE_PATHS_H_

#include <string>

namespace barbakan::core
{
	std::string getRepoFolder();
	std::string getBuildFolder();
	std::string getBuildDataFolder();
	std::string getBuildTestDataFolder();
}  // namespace barbakan::core

#endif  // BARBAKAN_CORE_PATHS_H_

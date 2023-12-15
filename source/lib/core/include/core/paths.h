#ifndef QCONV_CORE_PATHS_H_
#define QCONV_CORE_PATHS_H_

#include <string>

namespace qconv::core
{
	std::string getRepoFolder();
	std::string getBuildFolder();
	std::string getBuildDataFolder();
	std::string getBuildTestDataFolder();
}  // namespace qconv::core

#endif  // QCONV_CORE_PATHS_H_

#define STRINGIZE2(x) #x
#define STRINGIZE(x) STRINGIZE2(x)
#include <core/paths.h>

using namespace qconv;

std::string core::getRepoFolder()
{
#if defined (CORELIBFOLDERPATH)	
	std::string coreLibFolderPath = STRINGIZE(CORELIBFOLDERPATH);
	return coreLibFolderPath + "/../../../";
#else 
	throw("Core lib folder path should be given as prepocessor flags. Cmake should have configured it automatically.");
#endif
}

std::string core::getBuildFolder()
{
	std::string repoFolder = core::getRepoFolder();
#if defined (NDEBUG)
	return repoFolder + "build/release/";

#else
	return repoFolder + "build/debug/";
#endif
}

std::string core::getBuildDataFolder()
{
	std::string buildTypeFolder = core::getBuildFolder();
	return buildTypeFolder + "data/";
}

std::string core::getBuildTestDataFolder()
{
	std::string buildTypeFolder = core::getBuildFolder();
	return buildTypeFolder + "test/data/";
}

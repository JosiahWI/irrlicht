#include "CGLTFMeshFileLoader.h"
#include "coreutil.h"
#include "IAnimatedMesh.h"
#include "IReadFile.h"
#include "path.h"

namespace irr
{

namespace scene
{

CGLTFMeshFileLoader::CGLTFMeshFileLoader()
{
}

bool CGLTFMeshFileLoader::isALoadableFileExtension(
	const io::path& filename) const
{
	return core::hasFileExtension(filename, "gltf");
}

IAnimatedMesh* CGLTFMeshFileLoader::createMesh(io::IReadFile* file)
{
	return nullptr;
}

} // namespace irr

} // namespace scene


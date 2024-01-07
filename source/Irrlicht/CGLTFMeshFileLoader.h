#ifndef __C_GLTF_MESH_FILE_LOADER_INCLUDED__
#define __C_GLTF_MESH_FILE_LOADER_INCLUDED__

#include "CSkinnedMesh.h"
#include "IAnimatedMesh.h"
#include "IMeshLoader.h"
#include "IReadFile.h"
#include "irrTypes.h"
#include "path.h"
#include "S3DVertex.h"
#include "vector2d.h"
#include "vector3d.h"

#include <functional>
#include <tiniergltf.hpp>

#include <cstddef>
#include <vector>

namespace irr
{

namespace scene
{

class CGLTFMeshFileLoader : public IMeshLoader
{
public:
	CGLTFMeshFileLoader() noexcept;

	bool isALoadableFileExtension(const io::path& filename) const override;

	IAnimatedMesh* createMesh(io::IReadFile* file) override;

private:
	class BufferOffset
	{
	public:
		BufferOffset(const std::vector<unsigned char>& buf,
				const std::size_t offset);

		BufferOffset(const BufferOffset& other,
				const std::size_t fromOffset);

		unsigned char at(const std::size_t fromOffset) const;
	private:
		const std::vector<unsigned char>& m_buf;
		std::size_t m_offset;
		int m_filesize;
	};

	class MeshExtractor {
	public:
		using vertex_t = video::S3DVertex;

		MeshExtractor(const tiniergltf::GlTF& model) noexcept;

		MeshExtractor(const tiniergltf::GlTF&& model) noexcept;

		/* Gets indices for the given mesh/primitive.
		 *
		 * Values are return in Irrlicht winding order.
		 */
		std::optional<std::vector<u16>> getIndices(const std::size_t meshIdx,
				const std::size_t primitiveIdx) const;

		std::optional<std::vector<vertex_t>> getVertices(std::size_t meshIdx,
				const std::size_t primitiveIdx) const;

		std::size_t getMeshCount() const;

		std::size_t getPrimitiveCount(const std::size_t meshIdx) const;

		void load(CSkinnedMesh* mesh);

	private:
		tiniergltf::GlTF m_model;

		std::vector<std::function<void()>> m_mesh_loaders;

		std::vector<CSkinnedMesh::SJoint*> m_loaded_nodes;

		template <typename T>
		static T readPrimitive(const BufferOffset& readFrom);

		static core::vector2df readVec2DF(
				const BufferOffset& readFrom);

		/* Read a vec3df from a buffer with transformations applied.
		 *
		 * Values are returned in Irrlicht coordinates.
		 */
		static core::vector3df readVec3DF(
				const BufferOffset& readFrom);
		
		static core::matrix4 readMatrix4(
				const BufferOffset& readFrom);
		
		static core::quaternion readQuaternion(
				const BufferOffset& readFrom);

		void copyPositions(const std::size_t accessorIdx,
				std::vector<vertex_t>& vertices) const;

		void copyNormals(const std::size_t accessorIdx,
				std::vector<vertex_t>& vertices) const;

		void copyTCoords(const std::size_t accessorIdx,
				std::vector<vertex_t>& vertices) const;

		std::size_t getElemCount(const std::size_t accessorIdx) const;

		std::size_t getByteStride(const std::size_t accessorIdx) const;

		bool isAccessorNormalized(const std::size_t accessorIdx) const;

		BufferOffset getBuffer(const std::size_t accessorIdx) const;

		std::optional<std::size_t> getIndicesAccessorIdx(const std::size_t meshIdx,
				const std::size_t primitiveIdx) const;

		std::optional<std::size_t> getPositionAccessorIdx(const std::size_t meshIdx,
				const std::size_t primitiveIdx) const;

		/* Get the accessor id of the normals of a primitive.
		 */
		std::optional<std::size_t> getNormalAccessorIdx(const std::size_t meshIdx,
				const std::size_t primitiveIdx) const;

		/* Get the accessor id for the tcoords of a primitive.
		 */
		std::optional<std::size_t> getTCoordAccessorIdx(const std::size_t meshIdx,
				const std::size_t primitiveIdx) const;
		
		void deferAddMesh(
			const std::size_t meshIdx,
			const std::optional<std::size_t> skinIdx,
			CSkinnedMesh *mesh,
			CSkinnedMesh::SJoint *parentJoint);

		void loadNode(
			const std::size_t nodeIdx,
			CSkinnedMesh* mesh,
			CSkinnedMesh::SJoint *parentJoint);
		
		void loadNodes(CSkinnedMesh* mesh);

		void loadSkins(CSkinnedMesh* mesh);

		void loadAnimation(
			const std::size_t animIdx,
			CSkinnedMesh* mesh);
	};

	std::optional<tiniergltf::GlTF> tryParseGLTF(io::IReadFile* file);
};

} // namespace scene

} // namespace irr

#endif // __C_GLTF_MESH_FILE_LOADER_INCLUDED__


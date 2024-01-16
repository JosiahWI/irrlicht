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
#include <tuple>
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
	template<typename T>
	static T rawget(const void *ptr);

	template<class T>
	class Accessor {
		typedef std::variant<const u8*, std::vector<T>, std::tuple<>> Source; //, std::unique_ptr<T[]>, std::tuple<>> Source;
	public:
		static Accessor sparseIndices(
				const tiniergltf::GlTF& model,
				const tiniergltf::AccessorSparseIndices &indices,
				const std::size_t count);
		static Accessor sparseValues(
				const tiniergltf::GlTF& model,
				const tiniergltf::AccessorSparseValues &values,
				const std::size_t count,
				const std::size_t defaultByteStride);
		static Accessor base(
				const tiniergltf::GlTF& model,
				std::size_t accessorIdx);
		static Accessor make(const tiniergltf::GlTF& model, std::size_t accessorIdx);
		static constexpr tiniergltf::Accessor::Type getType();
		static constexpr tiniergltf::Accessor::ComponentType getComponentType();
		std::size_t getCount() const { return count; }
		T get(std::size_t i) const;
	private:
		Accessor(Source source, std::size_t byteStride, std::size_t count)
			: source(source), byteStride(byteStride), count(count) {}
		// HACK the 0 byte strides here aren't clean.
		Accessor(std::vector<T> vec, std::size_t count)
			: source(vec), byteStride(0), count(count) {}
		Accessor(std::size_t count)
			: source(std::make_tuple()), byteStride(0), count(count) {}
		// Directly from buffer, sparse, or default-initialized
		const Source source;
		const std::size_t byteStride;
		const std::size_t count;
	};

	template<typename... Ts>
	using AccessorVariant = std::variant<Accessor<Ts>...>;

	template<std::size_t N, typename... Ts>
	using ArrayAccessorVariant = std::variant<Accessor<std::array<Ts, N>>...>;

	template<std::size_t N>
	using NormalizedValuesAccessor = ArrayAccessorVariant<N, u8, u16, f32>;

	template<std::size_t N>
	static NormalizedValuesAccessor<N> createNormalizedValuesAccessor(
		const tiniergltf::GlTF& model,
		const std::size_t accessorIdx);

	template<std::size_t N>
	static std::array<f32, N> getNormalizedValues(
		const NormalizedValuesAccessor<N> &accessor,
		const std::size_t i);

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

		void copyPositions(const std::size_t accessorIdx,
				std::vector<vertex_t>& vertices) const;

		void copyNormals(const std::size_t accessorIdx,
				std::vector<vertex_t>& vertices) const;

		void copyTCoords(const std::size_t accessorIdx,
				std::vector<vertex_t>& vertices) const;
		
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


#include "CGLTFMeshFileLoader.h"
#include "CSkinnedMesh.h"
#include "coreutil.h"
#include "IAnimatedMesh.h"
#include "IReadFile.h"
#include "irrTypes.h"
#include "matrix4.h"
#include "path.h"
#include "S3DVertex.h"
#include "quaternion.h"
#include "tiniergltf.hpp"
#include "vector3d.h"
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

/* Notes on the coordinate system.
 *
 * glTF uses a right-handed coordinate system where +Z is the
 * front-facing axis, and Irrlicht uses a left-handed coordinate
 * system where -Z is the front-facing axis.
 * We convert between them by reflecting the mesh across the X axis.
 * Doing this correctly requires negating the Z coordinate on
 * vertex positions and normals, and reversing the winding order
 * of the vertex indices.
 */

namespace irr {

namespace scene {

CGLTFMeshFileLoader::BufferOffset::BufferOffset(
		const std::vector<unsigned char>& buf,
		const std::size_t offset)
	: m_buf(buf)
	, m_offset(offset)
{
}

CGLTFMeshFileLoader::BufferOffset::BufferOffset(
		const CGLTFMeshFileLoader::BufferOffset& other,
		const std::size_t fromOffset)
	: m_buf(other.m_buf)
	, m_offset(other.m_offset + fromOffset)
{
}

/**
 * Get a raw unsigned char (ubyte) from a buffer offset.
*/
unsigned char CGLTFMeshFileLoader::BufferOffset::at(
		const std::size_t fromOffset) const
{
	return m_buf.at(m_offset + fromOffset);
}

CGLTFMeshFileLoader::CGLTFMeshFileLoader() noexcept
{
}

/**
 * The most basic portion of the code base. This tells irllicht if this file has a .gltf extension.
*/
bool CGLTFMeshFileLoader::isALoadableFileExtension(
		const io::path& filename) const
{
	return core::hasFileExtension(filename, "gltf");
}

/**
 * Entry point into loading a GLTF model.
*/
IAnimatedMesh* CGLTFMeshFileLoader::createMesh(io::IReadFile* file)
{
	if (file->getSize() <= 0) {
		return nullptr;
	}
	std::optional<tiniergltf::GlTF> model = tryParseGLTF(file);
	if (!model.has_value()) {
		return nullptr;
	}

	if (!(model->buffers.has_value()
			&& model->bufferViews.has_value()
			&& model->accessors.has_value()
			&& model->meshes.has_value()
			&& model->nodes.has_value())) {
		return nullptr;
	}

	MeshExtractor parser(std::move(model.value()));
	CSkinnedMesh *mesh = new CSkinnedMesh();
	try {
		parser.loadNodes(mesh);
	} catch (std::runtime_error &e) {
		mesh->drop();
		return nullptr;
	}
	return mesh;
}


/**
 * Load up the rawest form of the model. The vertex positions and indices.
 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes
 * If material is undefined, then a default material MUST be used.
*/
void CGLTFMeshFileLoader::MeshExtractor::loadMesh(
		const std::size_t meshIdx,
		CSkinnedMesh *mesh,
		CSkinnedMesh::SJoint *parent) const
{
	for (std::size_t j = 0; j < getPrimitiveCount(meshIdx); ++j) {
		auto vertices = getVertices(meshIdx, j);
		if (!vertices.has_value())
			continue; // "When positions are not specified, client implementations SHOULD skip primitive’s rendering"

		// Excludes the max value for consistency.
		if (vertices->size() >= std::numeric_limits<u16>::max())
			throw std::runtime_error("too many vertices");

		// Apply the global transform along the parent chain.
		for (auto &vertex : *vertices) {
			parent->GlobalMatrix.transformVect(vertex.Pos);
			// Apply scaling, rotation and translation (in that order) to the normal.
			parent->GlobalMatrix.transformVect(vertex.Normal);
			// Undo the translation, leaving us with scaling and rotation.
			vertex.Normal -= parent->GlobalMatrix.getTranslation();
			// Renormalize (length might have been affected by scaling).
			vertex.Normal.normalize();
		}
		
		auto maybeIndices = getIndices(meshIdx, j);
		std::vector<u16> indices;
		if (maybeIndices.has_value()) {
			indices = std::move(maybeIndices.value());
			for (u16 index : indices) {
				if (index >= vertices->size())
					throw std::runtime_error("index out of bounds");
			}
		} else {
			// Non-indexed geometry
			indices = std::vector<u16>(vertices->size());
			for (u16 i = 0; i < vertices->size(); i++) {
				indices[i] = i;
			}
		}

		auto *meshbuf = mesh->addMeshBuffer();
		meshbuf->append(vertices->data(), vertices->size(),
			indices.data(), indices.size());
	}
}

// Base transformation between left & right handed coordinate systems.
// This just inverts the Z axis.
static core::matrix4 leftToRight = core::matrix4(
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, -1, 0,
	0, 0, 0, 1
);
static core::matrix4 rightToLeft = leftToRight;

static core::matrix4 loadTransform(std::optional<std::variant<tiniergltf::Node::Matrix, tiniergltf::Node::TRS>> transform) {
	if (!transform.has_value()) {
		return core::matrix4();
	}
	core::matrix4 mat;
	if (std::holds_alternative<tiniergltf::Node::Matrix>(*transform)) {
		const auto &m = std::get<tiniergltf::Node::Matrix>(*transform);
		// Note: Under the hood, this casts these doubles to floats.
		mat = core::matrix4(
			m[0], m[1], m[2], m[3],
			m[4], m[5], m[6], m[7],
			m[8], m[9], m[10], m[11],
			m[12], m[13], m[14], m[15]);
		// glTF uses column-major order, Irrlicht uses row-major order.
		mat = mat.getTransposed();
	} else {
		const auto &trs = std::get<tiniergltf::Node::TRS>(*transform);
		const auto &trans = trs.translation;
		const auto &rot = trs.rotation;
		const auto &scale = trs.scale;
		core::matrix4 transMat;
		transMat.setTranslation(core::vector3df(trans[0], trans[1], trans[2]));
		core::matrix4 rotMat = core::quaternion(rot[0], rot[1], rot[2], rot[3]).getMatrix();
		core::matrix4 scaleMat;
		scaleMat.setScale(core::vector3df(scale[0], scale[1], scale[2]));
		mat = transMat * rotMat * scaleMat;
	}
	return rightToLeft * mat * leftToRight;
}

void CGLTFMeshFileLoader::MeshExtractor::loadNode(
		const std::size_t nodeIdx,
		CSkinnedMesh* mesh,
		CSkinnedMesh::SJoint *parent) const
{
	const auto &node = m_model.nodes->at(nodeIdx);
	auto *joint = mesh->addJoint(parent);
	const core::matrix4 transform = loadTransform(node.transform);
	joint->LocalMatrix = transform;
	joint->GlobalMatrix = parent ? parent->GlobalMatrix * joint->LocalMatrix : joint->LocalMatrix;
	if (node.name.has_value()) {
		joint->Name = node.name->c_str();
	}
	if (node.mesh.has_value()) {
		loadMesh(*node.mesh, mesh, joint);
	}
	if (node.children.has_value()) {
		for (const auto &child : *node.children) {
			loadNode(child, mesh, joint);
		}
	}
}

void CGLTFMeshFileLoader::MeshExtractor::loadNodes(CSkinnedMesh* mesh) const
{
	std::vector<bool> isChild(m_model.nodes->size());
	for (const auto &node : *m_model.nodes) {
		if (!node.children.has_value())
			continue;
		for (const auto &child : *node.children) {
			isChild[child] = true;
		}
	}
	// Load all nodes that aren't children.
	// Children will be loaded by their parent nodes.
	for (std::size_t i = 0; i < m_model.nodes->size(); ++i) {
		if (!isChild[i]) {
			loadNode(i, mesh, nullptr);
		}
	}
}

CGLTFMeshFileLoader::MeshExtractor::MeshExtractor(
		const tiniergltf::GlTF& model) noexcept
	: m_model(model)
{
}

CGLTFMeshFileLoader::MeshExtractor::MeshExtractor(
		const tiniergltf::GlTF&& model) noexcept
	: m_model(model)
{
}

/**
 * Extracts GLTF mesh indices into the irrlicht model.
*/
std::optional<std::vector<u16>> CGLTFMeshFileLoader::MeshExtractor::getIndices(
		const std::size_t meshIdx,
		const std::size_t primitiveIdx) const
{
	const auto accessorIdx = getIndicesAccessorIdx(meshIdx, primitiveIdx);
	if (!accessorIdx.has_value())
		return std::nullopt; // non-indexed geometry
	const auto &accessor = m_model.accessors->at(accessorIdx.value());
	
	const auto& buf = getBuffer(accessorIdx.value());

	std::vector<u16> indices{};
	const auto count = getElemCount(accessorIdx.value());
	for (std::size_t i = 0; i < count; ++i) {
		std::size_t elemIdx = count - i - 1; // reverse index order
		u16 index;
		// Note: glTF forbids the max value for each component type.
		switch (accessor.componentType) {
			case tiniergltf::Accessor::ComponentType::UNSIGNED_BYTE: {
				index = readPrimitive<u8>(BufferOffset(buf, elemIdx * sizeof(u8)));
				if (index == std::numeric_limits<u8>::max())
					throw std::runtime_error("invalid index");
				break;
			}
			case tiniergltf::Accessor::ComponentType::UNSIGNED_SHORT: {
				index = readPrimitive<u16>(BufferOffset(buf, elemIdx * sizeof(u16)));
				if (index == std::numeric_limits<u16>::max())
					throw std::runtime_error("invalid index");
				break;
			}
			case tiniergltf::Accessor::ComponentType::UNSIGNED_INT: {
				u32 indexWide = readPrimitive<u32>(BufferOffset(buf, elemIdx * sizeof(u32)));
				// Use >= here for consistency.
				if (indexWide >= std::numeric_limits<u16>::max())
					throw std::runtime_error("index too large (>= 65536)");
				index = (u16) indexWide;
				break;
			}
			default:
				throw std::runtime_error("invalid index component type");
		}
		indices.push_back(index);
	}

	return indices;
}

/**
 * Create a vector of video::S3DVertex (model data) from a mesh & primitive index.
*/
std::optional<std::vector<video::S3DVertex>> CGLTFMeshFileLoader::MeshExtractor::getVertices(
		const std::size_t meshIdx,
		const std::size_t primitiveIdx) const
{
	const auto positionAccessorIdx = getPositionAccessorIdx(
			meshIdx, primitiveIdx);
	if (!positionAccessorIdx.has_value()) {
		// "When positions are not specified, client implementations SHOULD skip primitive's rendering"
		return std::nullopt;
	}

	std::vector<vertex_t> vertices{};
	vertices.resize(getElemCount(*positionAccessorIdx));
	copyPositions(*positionAccessorIdx, vertices);

	const auto normalAccessorIdx = getNormalAccessorIdx(
			meshIdx, primitiveIdx);
	if (normalAccessorIdx.has_value()) {
		copyNormals(normalAccessorIdx.value(), vertices);
	}

	const auto tCoordAccessorIdx = getTCoordAccessorIdx(
			meshIdx, primitiveIdx);
	if (tCoordAccessorIdx.has_value()) {
		copyTCoords(tCoordAccessorIdx.value(), vertices);
	}

	return vertices;
}

/**
 * Get the amount of meshes that a model contains.
*/
std::size_t CGLTFMeshFileLoader::MeshExtractor::getMeshCount() const
{
	return m_model.meshes->size();
}

/**
 * Get the amount of primitives that a mesh in a model contains.
*/
std::size_t CGLTFMeshFileLoader::MeshExtractor::getPrimitiveCount(
		const std::size_t meshIdx) const
{
	return m_model.meshes->at(meshIdx).primitives.size();
}

/**
 * Templated buffer reader. Based on type width.
 * This is specifically used to build upon to read more complex data types.
 * It is also used raw to read arrays directly.
 * Basically we're using the width of the type to infer 
 * how big of a gap we have from the beginning of the buffer.
*/
template <typename T>
T CGLTFMeshFileLoader::MeshExtractor::readPrimitive(
		const BufferOffset& readFrom)
{
	unsigned char d[sizeof(T)]{};
	for (std::size_t i = 0; i < sizeof(T); ++i) {
		d[i] = readFrom.at(i);
	}
	T dest;
	std::memcpy(&dest, d, sizeof(dest));
	return dest;
}

/**
 * Read a vector2df from a buffer at an offset.
 * @return vec2 core::Vector2df
*/
core::vector2df CGLTFMeshFileLoader::MeshExtractor::readVec2DF(
		const CGLTFMeshFileLoader::BufferOffset& readFrom)
{
	return core::vector2df(readPrimitive<float>(readFrom),
		readPrimitive<float>(BufferOffset(readFrom, sizeof(float))));

}

/**
 * Read a vector3df from a buffer at an offset.
 * Also does right-to-left-handed coordinate system conversion (inverts Z axis).
 * @return vec3 core::Vector3df
*/
core::vector3df CGLTFMeshFileLoader::MeshExtractor::readVec3DF(
		const BufferOffset& readFrom,
		const core::vector3df scale = {1.0f,1.0f,1.0f})
{
	return core::vector3df(
		readPrimitive<float>(readFrom),
		readPrimitive<float>(BufferOffset(readFrom, sizeof(float))),
		-readPrimitive<float>(BufferOffset(readFrom, 2 *
		sizeof(float))));
}

/**
 * Streams vertex positions raw data into usable buffer via reference.
 * Buffer: ref Vector<video::S3DVertex>
*/
void CGLTFMeshFileLoader::MeshExtractor::copyPositions(
		const std::size_t accessorIdx,
		std::vector<vertex_t>& vertices) const
{

	const auto& buffer = getBuffer(accessorIdx);
	const auto count = getElemCount(accessorIdx);
	const auto byteStride = getByteStride(accessorIdx);

	for (std::size_t i = 0; i < count; i++) {
		const auto v = readVec3DF(BufferOffset(buffer, byteStride * i));
		vertices[i].Pos = v;
	}
}

/**
 * Streams normals raw data into usable buffer via reference.
 * Buffer: ref Vector<video::S3DVertex>
*/
void CGLTFMeshFileLoader::MeshExtractor::copyNormals(
		const std::size_t accessorIdx,
		std::vector<vertex_t>& vertices) const
{
	const auto& buffer = getBuffer(accessorIdx);
	const auto count = getElemCount(accessorIdx);
	
	for (std::size_t i = 0; i < count; i++) {
		const auto n = readVec3DF(BufferOffset(buffer,
			3 * sizeof(float) * i));
		vertices[i].Normal = n;
	}
}

/**
 * Streams texture coordinate raw data into usable buffer via reference.
 * Buffer: ref Vector<video::S3DVertex>
*/
void CGLTFMeshFileLoader::MeshExtractor::copyTCoords(
		const std::size_t accessorIdx,
		std::vector<vertex_t>& vertices) const
{

	const auto& buffer = getBuffer(accessorIdx);
	const auto count = getElemCount(accessorIdx);

	for (std::size_t i = 0; i < count; ++i) {
		const auto t = readVec2DF(BufferOffset(buffer,
			2 * sizeof(float) * i));
		vertices[i].TCoords = t;
	}
}

/**
 * The number of elements referenced by this accessor, not to be confused with the number of bytes or number of components.
 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#_accessor_count
 * Type: Integer
 * Required: YES
*/
std::size_t CGLTFMeshFileLoader::MeshExtractor::getElemCount(
		const std::size_t accessorIdx) const
{
	return m_model.accessors->at(accessorIdx).count;
}

/**
 * The stride, in bytes, between vertex attributes.
 * When this is not defined, data is tightly packed.
 * When two or more accessors use the same buffer view, this field MUST be defined.
 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#_bufferview_bytestride
 * Required: NO
*/
std::size_t CGLTFMeshFileLoader::MeshExtractor::getByteStride(
		const std::size_t accessorIdx) const
{
	const auto& accessor = m_model.accessors->at(accessorIdx);
	// FIXME this does not work with sparse / zero-initialized accessors
	const auto& view = m_model.bufferViews->at(accessor.bufferView.value());
	return view.byteStride.value_or(accessor.elementSize());
}

/**
 * Specifies whether integer data values are normalized (true) to [0, 1] (for unsigned types) 
 * or to [-1, 1] (for signed types) when they are accessed. This property MUST NOT be set to
 * true for accessors with FLOAT or UNSIGNED_INT component type.
 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#_accessor_normalized
 * Required: NO
*/
bool CGLTFMeshFileLoader::MeshExtractor::isAccessorNormalized(
	const std::size_t accessorIdx) const
{
	const auto& accessor = m_model.accessors->at(accessorIdx);
	return accessor.normalized;
}

/**
 * Walk through the complex chain of the model to extract the required buffer.
 * Accessor -> BufferView -> Buffer
*/
CGLTFMeshFileLoader::BufferOffset CGLTFMeshFileLoader::MeshExtractor::getBuffer(
		const std::size_t accessorIdx) const
{
	const auto& accessor = m_model.accessors->at(accessorIdx);
	// FIXME this does not work with sparse / zero-initialized accessors
	const auto& view = m_model.bufferViews->at(accessor.bufferView.value());
	const auto& buffer = m_model.buffers->at(view.buffer);

	return BufferOffset(buffer.data, view.byteOffset);
}

/**
 * The index of the accessor that contains the vertex indices. 
 * When this is undefined, the primitive defines non-indexed geometry. 
 * When defined, the accessor MUST have SCALAR type and an unsigned integer component type.
 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#_mesh_primitive_indices
 * Type: Integer
 * Required: NO
*/
std::optional<std::size_t> CGLTFMeshFileLoader::MeshExtractor::getIndicesAccessorIdx(
		const std::size_t meshIdx,
		const std::size_t primitiveIdx) const
{
	return m_model.meshes->at(meshIdx).primitives[primitiveIdx].indices;
}

/**
 * The index of the accessor that contains the POSITIONs.
 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes-overview
 * Type: VEC3 (Float)
*/
std::optional<std::size_t> CGLTFMeshFileLoader::MeshExtractor::getPositionAccessorIdx(
		const std::size_t meshIdx,
		const std::size_t primitiveIdx) const
{
	return m_model.meshes->at(meshIdx).primitives[primitiveIdx].attributes.position;
}

/**
 * The index of the accessor that contains the NORMALs.
 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes-overview
 * Type: VEC3 (Float)
 * ! Required: NO (Appears to not be, needs another pair of eyes to research.)
*/
std::optional<std::size_t> CGLTFMeshFileLoader::MeshExtractor::getNormalAccessorIdx(
		const std::size_t meshIdx,
		const std::size_t primitiveIdx) const
{
	return m_model.meshes->at(meshIdx).primitives[primitiveIdx].attributes.normal;
}

/**
 * The index of the accessor that contains the TEXCOORDs.
 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes-overview
 * Type: VEC3 (Float)
 * ! Required: YES (Appears so, needs another pair of eyes to research.)
*/
std::optional<std::size_t> CGLTFMeshFileLoader::MeshExtractor::getTCoordAccessorIdx(
		const std::size_t meshIdx,
		const std::size_t primitiveIdx) const
{
	const auto& texcoords = m_model.meshes->at(meshIdx).primitives[primitiveIdx].attributes.texcoord;
	if (!texcoords.has_value())
		return std::nullopt;
	return texcoords->at(0);
}

/**
 * This is where the actual model's GLTF file is loaded and parsed by tiniergltf.
*/
std::optional<tiniergltf::GlTF> CGLTFMeshFileLoader::tryParseGLTF(io::IReadFile* file)
{
	auto size = file->getSize();
	auto buf = std::make_unique<char[]>(size + 1);
	file->read(buf.get(), size);
	// We probably don't need this, but add it just to be sure.
	buf[size] = '\0';
	Json::CharReaderBuilder builder;
    const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
	Json::Value json;
	JSONCPP_STRING err;
    if (!reader->parse(buf.get(), buf.get() + size, &json, &err)) {
      return std::nullopt;
    }
	try {
		return tiniergltf::GlTF(json);
	}  catch (const std::runtime_error &e) {
		return std::nullopt;
	} catch (const std::out_of_range &e) {
		return std::nullopt;
	}
}

} // namespace scene

} // namespace irr


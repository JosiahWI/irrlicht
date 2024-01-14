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

// Right-to-left handedness conversions

template<typename T>
static inline T convertHandedness(const T &t);

template<>
core::vector3df convertHandedness(const core::vector3df &p) {
	return core::vector3df(p.X, p.Y, -p.Z);
}

template<>
core::quaternion convertHandedness(const core::quaternion &q) {
	return core::quaternion(q.X, q.Y, -q.Z, q.W);
}

template<>
core::matrix4 convertHandedness(const core::matrix4 &mat) {
	// Base transformation between left & right handed coordinate systems.
	static core::matrix4 invertZ = core::matrix4(
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, 1
	);
	// Convert from left-handed to right-handed,
	// then apply mat,
	// then convert from right-handed to left-handed.
	// Both conversions just invert Z.
	return invertZ * mat * invertZ;
}

template<class T>
CGLTFMeshFileLoader::Accessor<T>::Accessor(const tiniergltf::GlTF& model, std::size_t accessorIdx)
{
	const auto& accessor = model.accessors->at(accessorIdx);
	const auto& view = model.bufferViews->at(accessor.bufferView.value());
	byteStride = view.byteStride.value_or(accessor.elementSize());
	if (accessor.componentType != getComponentType() || accessor.type != getType())
		throw std::runtime_error("invalid accessor");
	count = accessor.count;

	const auto& buffer = model.buffers->at(view.buffer);
	buf = buffer.data.data() + view.byteOffset + accessor.byteOffset;
}

#define ACCESSOR_TYPES(T, U, V) \
	template<> \
	tiniergltf::Accessor::Type CGLTFMeshFileLoader::Accessor<T>::getType() { \
		return tiniergltf::Accessor::Type::U; \
	} \
	template<> \
	tiniergltf::Accessor::ComponentType CGLTFMeshFileLoader::Accessor<T>::getComponentType() { \
		return tiniergltf::Accessor::ComponentType::V; \
	} \

#define VEC_ACCESSOR_TYPES(T, U, n) \
	template<> \
	tiniergltf::Accessor::Type CGLTFMeshFileLoader::Accessor<std::array<T, n>>::getType() { \
		return tiniergltf::Accessor::Type::VEC##n; \
	} \
	template<> \
	tiniergltf::Accessor::ComponentType CGLTFMeshFileLoader::Accessor<std::array<T, n>>::getComponentType() { \
		return tiniergltf::Accessor::ComponentType::U; \
	} \
	template<> \
	std::array<T, n> CGLTFMeshFileLoader::rawget(const void *ptr) { \
		const T *tptr = reinterpret_cast<const T*>(ptr); \
		std::array<T, n> res; \
		for (u8 i = 0; i < n; ++i) \
			res[i] = rawget<T>(tptr + i); \
		return res; \
	}

#define ACCESSOR_PRIMITIVE(T, U) ACCESSOR_TYPES(T, SCALAR, U) \
	VEC_ACCESSOR_TYPES(T, U, 2) \
	VEC_ACCESSOR_TYPES(T, U, 3) \
	VEC_ACCESSOR_TYPES(T, U, 4) \

ACCESSOR_PRIMITIVE(f32, FLOAT)
ACCESSOR_PRIMITIVE(u8, UNSIGNED_BYTE)
ACCESSOR_PRIMITIVE(u16, UNSIGNED_SHORT)
ACCESSOR_PRIMITIVE(u32, UNSIGNED_INT)

ACCESSOR_TYPES(core::vector2df, VEC2, FLOAT)
ACCESSOR_TYPES(core::vector3df, VEC3, FLOAT)
ACCESSOR_TYPES(core::quaternion, VEC4, FLOAT)
ACCESSOR_TYPES(core::matrix4, MAT4, FLOAT)

template<class T>
T CGLTFMeshFileLoader::Accessor<T>::get(std::size_t i) const
{
	return rawget<T>(buf + i * byteStride);
}

// Note: clang and gcc should both optimize this out.
static inline bool isBigEndian() {
	const u16 x = 0xFF00;
	return *(const u8*)(&x);
}

template<typename T>
T CGLTFMeshFileLoader::rawget(const void *ptr) {
	if (!isBigEndian())
		return *reinterpret_cast<const T*>(ptr);
	// glTF uses little endian.
	// On big-endian systems, we have to swap the byte order.
	// TODO test this
	const u8 *bptr = reinterpret_cast<const u8*>(ptr);
	u8 bytes[sizeof(T)];
	for (std::size_t i = 0; i < sizeof(T); ++i) {
		bytes[sizeof(T) - i - 1] = bptr[i];
	}
	return *reinterpret_cast<const T*>(bytes);
}

// Note that these "more specialized templates" should win.

template<>
core::matrix4 CGLTFMeshFileLoader::rawget(const void *ptr) {
	const f32 *fptr = reinterpret_cast<const f32*>(ptr);
	f32 M[16];
	for (u8 i = 0; i < 16; ++i) {
		M[i] = rawget<f32>(fptr + i);
	}
	core::matrix4 mat;
	mat.setM(M);
	return mat;
}

template<>
core::vector2df CGLTFMeshFileLoader::rawget(const void *ptr) {
	const f32 *fptr = reinterpret_cast<const f32*>(ptr);
	return core::vector2df(
		rawget<f32>(fptr),
		rawget<f32>(fptr + 1));
}

template<>
core::vector3df CGLTFMeshFileLoader::rawget(const void *ptr) {
	const f32 *fptr = reinterpret_cast<const f32*>(ptr);
	return core::vector3df(
		rawget<f32>(fptr),
		rawget<f32>(fptr + 1),
		rawget<f32>(fptr + 2));
}

template<>
core::quaternion CGLTFMeshFileLoader::rawget(const void *ptr) {
	const f32 *fptr = reinterpret_cast<const f32*>(ptr);
	return core::quaternion(
		rawget<f32>(fptr),
		rawget<f32>(fptr + 1),
		rawget<f32>(fptr + 2), 
		rawget<f32>(fptr + 3));
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
		parser.load(mesh);
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
void CGLTFMeshFileLoader::MeshExtractor::deferAddMesh(
		const std::size_t meshIdx,
		const std::optional<std::size_t> skinIdx,
		CSkinnedMesh *mesh,
		CSkinnedMesh::SJoint *parent)
{
	m_mesh_loaders.push_back([=] {
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
			const auto buffer_id = mesh->getMeshBufferCount() - 1;
			
			if (!skinIdx.has_value())
				continue;
			const auto &skin = m_model.skins->at(*skinIdx);

			const auto &attrs = m_model.meshes->at(meshIdx).primitives.at(j).attributes;
			const auto &joints = attrs.joints;
			if (!joints.has_value())
				continue;

			const auto &weights = attrs.weights;
			for (std::size_t set = 0; set < joints->size(); ++set) {
				const auto jointAccessor = ([&]() -> std::variant<Accessor<std::array<u8, 4>>, Accessor<std::array<u16, 4>>> {
					const auto idx = joints->at(set);
					const auto &acc = m_model.accessors->at(idx);
					if (acc.type != tiniergltf::Accessor::Type::VEC4)
						throw std::runtime_error("invalid accessor type");

					switch (acc.componentType) {
						case tiniergltf::Accessor::ComponentType::UNSIGNED_BYTE:
							return Accessor<std::array<u8, 4>>(m_model, idx);
						case tiniergltf::Accessor::ComponentType::UNSIGNED_SHORT:
							return Accessor<std::array<u16, 4>>(m_model, idx);
						default:
							throw std::runtime_error("invalid component type");
					}
				})();

				const auto weightAccessor = ([&]() -> std::variant<
						Accessor<std::array<u8, 4>>,
						Accessor<std::array<u16, 4>>,
						Accessor<std::array<f32, 4>>> {
					const auto idx = weights->at(set);
					const auto &acc = m_model.accessors->at(idx);
					if (acc.type != tiniergltf::Accessor::Type::VEC4)
						throw std::runtime_error("invalid accessor type");

					switch (acc.componentType) {
						case tiniergltf::Accessor::ComponentType::UNSIGNED_BYTE:
							return Accessor<std::array<u8, 4>>(m_model, idx);
						case tiniergltf::Accessor::ComponentType::UNSIGNED_SHORT:
							return Accessor<std::array<u16, 4>>(m_model, idx);
						case tiniergltf::Accessor::ComponentType::FLOAT:
							return Accessor<std::array<f32, 4>>(m_model, idx);
						default:
							throw std::runtime_error("invalid component type");
					}
				})();

				for (std::size_t v = 0; v < vertices->size(); ++v) {
					std::array<u16, 4> jointIdxs;
					if (std::holds_alternative<Accessor<std::array<u8, 4>>>(jointAccessor)) {
						const auto jointIdxsU8 = std::get<Accessor<std::array<u8, 4>>>(jointAccessor).get(v);
						jointIdxs = {jointIdxsU8[0], jointIdxsU8[1], jointIdxsU8[2], jointIdxsU8[3]};
					} else if (std::holds_alternative<Accessor<std::array<u16, 4>>>(jointAccessor)) {
						jointIdxs = std::get<Accessor<std::array<u16, 4>>>(jointAccessor).get(v);
					}
					std::array<f32, 4> strengths;
					if (std::holds_alternative<Accessor<std::array<u8, 4>>>(weightAccessor)) {
						const auto u8s = std::get<Accessor<std::array<u8, 4>>>(weightAccessor).get(v);
						for (u8 i = 0; i < 4; ++i)
							strengths[i] = static_cast<f32>(u8s[i]) / std::numeric_limits<u8>::max();
					} else if (std::holds_alternative<Accessor<std::array<u16, 4>>>(weightAccessor)) {
						const auto u16s = std::get<Accessor<std::array<u16, 4>>>(weightAccessor).get(v);
						for (u8 i = 0; i < 4; ++i)
							strengths[i] = static_cast<f32>(u16s[i]) / std::numeric_limits<u16>::max();
					} else {
						strengths = std::get<Accessor<std::array<f32, 4>>>(weightAccessor).get(v);
					}

					// 4 joints per set
					for (std::size_t in_set = 0; in_set < 4; ++in_set) {
						u16 jointIdx = jointIdxs[in_set];
						f32 strength = strengths[in_set];
						if (strength == 0)
							continue;

						CSkinnedMesh::SWeight *weight = mesh->addWeight(m_loaded_nodes.at(skin.joints.at(jointIdx)));
						weight->buffer_id = buffer_id;
						weight->vertex_id = v;
						weight->strength = strength;
					}
				}
			}
		}
	});
}

static core::matrix4 loadTransform(std::optional<std::variant<tiniergltf::Node::Matrix, tiniergltf::Node::TRS>> transform, CSkinnedMesh::SJoint *joint) {
	if (!transform.has_value()) {
		return core::matrix4();
	}
	if (std::holds_alternative<tiniergltf::Node::Matrix>(*transform)) {
		// TODO test this path using glTF sample models
		const auto &m = std::get<tiniergltf::Node::Matrix>(*transform);
		// Note: Under the hood, this casts these doubles to floats.
		core::matrix4 mat = convertHandedness(core::matrix4(
			m[0], m[1], m[2], m[3],
			m[4], m[5], m[6], m[7],
			m[8], m[9], m[10], m[11],
			m[12], m[13], m[14], m[15]));

		// Decompose the matrix into translation, scale, and rotation.
		joint->Animatedposition = mat.getTranslation();

		auto scale = mat.getScale();
		joint->Animatedscale = scale;
		core::matrix4 inverseScale;
		inverseScale.setScale(core::vector3df(
			scale.X == 0 ? 0 : 1/scale.X,
			scale.Y == 0 ? 0 : 1/scale.Y,
			scale.Z == 0 ? 0 : 1/scale.Z
		));

		core::matrix4 axisNormalizedMat = inverseScale * mat;
		joint->Animatedrotation = axisNormalizedMat.getRotationDegrees();
		// Invert the rotation because it is applied using `getMatrix_transposed`,
		// which again inverts.
		joint->Animatedrotation.makeInverse();
		
		return mat;
	} else {
		const auto &trs = std::get<tiniergltf::Node::TRS>(*transform);
		const auto &trans = trs.translation;
		const auto &rot = trs.rotation;
		const auto &scale = trs.scale;
		core::matrix4 transMat;
		joint->Animatedposition = core::vector3df(trans[0], trans[1], -trans[2]);
		transMat.setTranslation(joint->Animatedposition);
		core::matrix4 rotMat;
		joint->Animatedrotation = convertHandedness(core::quaternion(rot[0], rot[1], rot[2], rot[3]));
		core::quaternion(joint->Animatedrotation).getMatrix_transposed(rotMat);
		joint->Animatedscale = core::vector3df(scale[0], scale[1], scale[2]);
		core::matrix4 scaleMat;
		scaleMat.setScale(joint->Animatedscale);
		return transMat * rotMat * scaleMat;
	}
}

void CGLTFMeshFileLoader::MeshExtractor::loadNode(
		const std::size_t nodeIdx,
		CSkinnedMesh* mesh,
		CSkinnedMesh::SJoint *parent)
{
	const auto &node = m_model.nodes->at(nodeIdx);
	auto *joint = mesh->addJoint(parent);
	const core::matrix4 transform = loadTransform(node.transform, joint);
	joint->LocalMatrix = transform;
	
	joint->GlobalMatrix = parent ? parent->GlobalMatrix * joint->LocalMatrix : joint->LocalMatrix;
	if (node.name.has_value()) {
		joint->Name = node.name->c_str();
	}
	m_loaded_nodes[nodeIdx] = joint;
	if (node.mesh.has_value()) {
		deferAddMesh(*node.mesh, node.skin, mesh, joint);
	}
	if (node.children.has_value()) {
		for (const auto &child : *node.children) {
			loadNode(child, mesh, joint);
		}
	}
}

void CGLTFMeshFileLoader::MeshExtractor::loadNodes(CSkinnedMesh* mesh)
{
	m_loaded_nodes = std::vector<CSkinnedMesh::SJoint*>(m_model.nodes->size());

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

void CGLTFMeshFileLoader::MeshExtractor::loadSkins(CSkinnedMesh* mesh)
{
	if (!m_model.skins.has_value())
		return;

	for (const auto &skin : *m_model.skins) {
		if (!skin.inverseBindMatrices.has_value())
			continue;
		const Accessor<core::matrix4> accessor(m_model, *skin.inverseBindMatrices);
		if (accessor.getCount() < skin.joints.size())
			throw std::runtime_error("accessor contains too few matrices");
		for (std::size_t i = 0; i < skin.joints.size(); ++i) {
			m_loaded_nodes.at(skin.joints[i])->GlobalInversedMatrix = convertHandedness(accessor.get(i));
		}
	}
}

void CGLTFMeshFileLoader::MeshExtractor::loadAnimation(
	const std::size_t animIdx, CSkinnedMesh* mesh)
{
	const auto &anim = m_model.animations->at(animIdx);
	for (const auto &channel : anim.channels) {

		const auto &sampler = anim.samplers.at(channel.sampler);
		if (sampler.interpolation != tiniergltf::AnimationSampler::Interpolation::LINEAR)
			throw std::runtime_error("unsupported interpolation");

		const Accessor<f32> inputAccessor(m_model, sampler.input);
		const auto n_frames = inputAccessor.getCount();

		if (!channel.target.node.has_value())
			throw std::runtime_error("no animated node");

		const auto &joint = m_loaded_nodes.at(*channel.target.node);
		switch (channel.target.path) {
			case tiniergltf::AnimationChannelTarget::Path::TRANSLATION: {
				const Accessor<core::vector3df> outputAccessor(m_model, sampler.output);
				for (std::size_t i = 0; i < n_frames; ++i) {
					auto *key = mesh->addPositionKey(joint);
					key->frame = inputAccessor.get(i);
					key->position = convertHandedness(outputAccessor.get(i));
				}
				break;
			}
			case tiniergltf::AnimationChannelTarget::Path::ROTATION: {
				const Accessor<core::quaternion> outputAccessor(m_model, sampler.output);
				for (std::size_t i = 0; i < n_frames; ++i) {
					auto *key = mesh->addRotationKey(joint);
					key->frame = inputAccessor.get(i);
					key->rotation = convertHandedness(outputAccessor.get(i));
				}
				break;
			}
			case tiniergltf::AnimationChannelTarget::Path::SCALE: {
				const Accessor<core::vector3df> outputAccessor(m_model, sampler.output);
				for (std::size_t i = 0; i < n_frames; ++i) {
					auto *key = mesh->addScaleKey(joint);
					key->frame = inputAccessor.get(i);
					key->scale = outputAccessor.get(i);
				}
				break;
			}
			case tiniergltf::AnimationChannelTarget::Path::WEIGHTS:
				throw std::runtime_error("no support for morph animations");
		}
	}
}

void CGLTFMeshFileLoader::MeshExtractor::load(CSkinnedMesh* mesh)
{
	loadNodes(mesh);
	for (const auto &loadMesh : m_mesh_loaders) {
		loadMesh();
	}
	loadSkins(mesh);
	// Load the first animation, if there is one.
	// Minetest does not support multiple animations yet.
	if (m_model.animations.has_value()) {
		loadAnimation(0, mesh);
		mesh->setAnimationSpeed(1);
	}
	mesh->finalize();
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
	const auto accessorIdx = m_model.meshes->at(meshIdx).primitives.at(primitiveIdx).indices;
	if (!accessorIdx.has_value())
		return std::nullopt; // non-indexed geometry

	const auto accessor = ([&]() -> std::variant<Accessor<u8>, Accessor<u16>, Accessor<u32>> {
		const auto &acc = m_model.accessors->at(*accessorIdx);
		switch (acc.componentType) {
			case tiniergltf::Accessor::ComponentType::UNSIGNED_BYTE:
				return Accessor<u8>(m_model, *accessorIdx);
			case tiniergltf::Accessor::ComponentType::UNSIGNED_SHORT:
				return Accessor<u16>(m_model, *accessorIdx);
			case tiniergltf::Accessor::ComponentType::UNSIGNED_INT:
				return Accessor<u32>(m_model, *accessorIdx);
			default:
				throw std::runtime_error("invalid component type");
		}
	})();
	const auto count = std::visit([](auto&& a) {return a.getCount();}, accessor);

	std::vector<u16> indices;
	for (std::size_t i = 0; i < count; ++i) {
		// TODO (low-priority) also reverse winding order based on determinant of global transform
		// FIXME this hack also reverses triangle draw order
		std::size_t elemIdx = count - i - 1; // reverse index order
		u16 index;
		// Note: glTF forbids the max value for each component type.
		if (std::holds_alternative<Accessor<u8>>(accessor)) {
			index = std::get<Accessor<u8>>(accessor).get(elemIdx);
			if (index == std::numeric_limits<u8>::max())
				throw std::runtime_error("invalid index");
		} else if (std::holds_alternative<Accessor<u16>>(accessor)) {
			index = std::get<Accessor<u16>>(accessor).get(elemIdx);
			if (index == std::numeric_limits<u16>::max())
				throw std::runtime_error("invalid index");
		} else if (std::holds_alternative<Accessor<u32>>(accessor)) {
			u32 indexWide = std::get<Accessor<u32>>(accessor).get(elemIdx);
			// Use >= here for consistency.
			if (indexWide >= std::numeric_limits<u16>::max())
				throw std::runtime_error("index too large (>= 65536)");
			index = indexWide;
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
	const auto &attributes = m_model.meshes->at(meshIdx).primitives.at(primitiveIdx).attributes;
	const auto positionAccessorIdx = attributes.position;
	if (!positionAccessorIdx.has_value()) {
		// "When positions are not specified, client implementations SHOULD skip primitive's rendering"
		return std::nullopt;
	}

	std::vector<vertex_t> vertices{};
	// HACK peeking directly in the accessor data doesn't feel right
	const auto vertexCount = m_model.accessors->at(*positionAccessorIdx).count;
	vertices.resize(vertexCount);
	copyPositions(*positionAccessorIdx, vertices);

	const auto normalAccessorIdx = attributes.normal;
	if (normalAccessorIdx.has_value()) {
		copyNormals(normalAccessorIdx.value(), vertices);
	}
	// TODO verify that the automatic normal recalculation done in Minetest indeed works correctly

	const auto& texcoords = m_model.meshes->at(meshIdx).primitives[primitiveIdx].attributes.texcoord;
	if (texcoords.has_value()) {
		const auto tCoordAccessorIdx = texcoords->at(0);
		copyTCoords(tCoordAccessorIdx, vertices);
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
 * Streams vertex positions raw data into usable buffer via reference.
 * Buffer: ref Vector<video::S3DVertex>
*/
void CGLTFMeshFileLoader::MeshExtractor::copyPositions(
		const std::size_t accessorIdx,
		std::vector<vertex_t>& vertices) const
{
	const Accessor<core::vector3df> accessor(m_model, accessorIdx);
	for (std::size_t i = 0; i < accessor.getCount(); i++) {
		vertices[i].Pos = convertHandedness(accessor.get(i));
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
	const Accessor<core::vector3df> accessor(m_model, accessorIdx);
	for (std::size_t i = 0; i < accessor.getCount(); i++) {
		vertices[i].Normal = convertHandedness(accessor.get(i));
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
	const Accessor<core::vector2df> accessor(m_model, accessorIdx);
	for (std::size_t i = 0; i < accessor.getCount(); i++) {
		vertices[i].TCoords = accessor.get(i);
	}
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


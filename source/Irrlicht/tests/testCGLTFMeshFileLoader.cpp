#include "CReadFile.h"
#include "CSkinnedMesh.h"
#include "matrix4.h"
#include "quaternion.h"
#include "vector3d.h"

#include <irrlicht.h>
#include <array>
#include <stdexcept>

// Catch needs to be included after Irrlicht so that it sees operator<<
// declarations.
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

using v3f = irr::core::vector3df;

class ScopedMesh
{
public:
	ScopedMesh(irr::io::IReadFile *file) :
			m_device{irr::createDevice(irr::video::EDT_NULL)}, m_mesh{nullptr}
	{
		auto *smgr = m_device->getSceneManager();
		m_mesh = smgr->getMesh(file);
	}

	ScopedMesh(const irr::io::path &filepath) :
			m_device{irr::createDevice(irr::video::EDT_NULL)}, m_mesh{nullptr}
	{
		auto *smgr = m_device->getSceneManager();
		irr::io::CReadFile f = irr::io::CReadFile(filepath);
		m_mesh = smgr->getMesh(&f);
	}

	~ScopedMesh()
	{
		m_device->drop();
		m_mesh = nullptr;
	}

	const irr::scene::IAnimatedMesh *getMesh() const
	{
		return m_mesh;
	}

private:
	irr::IrrlichtDevice *m_device;
	irr::scene::IAnimatedMesh *m_mesh;
};

TEST_CASE("error cases")
{
	SECTION("null file pointer")
	{
		ScopedMesh sm(nullptr);
		CHECK(sm.getMesh() == nullptr);
	}
	SECTION("empty file")
	{
		ScopedMesh sm("source/Irrlicht/tests/assets/invalid/empty.gltf");
		CHECK(sm.getMesh() == nullptr);
	}
	SECTION("invalid JSON")
	{
		ScopedMesh sm("source/Irrlicht/tests/assets/invalid/json_missing_brace.gltf");
		CHECK(sm.getMesh() == nullptr);
	}
	// This is an example of something that should be validated by tiniergltf.
	SECTION("invalid bufferview bounds")
	{
		ScopedMesh sm("source/Irrlicht/tests/assets/invalid/invalid_bufferview_bounds.gltf");
		CHECK(sm.getMesh() == nullptr);
	}
}

TEST_CASE("minimal triangle")
{
	auto path = GENERATE(
			"source/Irrlicht/tests/assets/minimal_triangle.gltf",
			"source/Irrlicht/tests/assets/triangle_with_vertex_stride.gltf");
	INFO(path);
	ScopedMesh sm(path);
	REQUIRE(sm.getMesh() != nullptr);
	REQUIRE(sm.getMesh()->getMeshBufferCount() == 1);

	SECTION("vertex coordinates are correct")
	{
		REQUIRE(sm.getMesh()->getMeshBuffer(0)->getVertexCount() == 3);
		const auto *vertices = reinterpret_cast<irr::video::S3DVertex *>(
				sm.getMesh()->getMeshBuffer(0)->getVertices());
		CHECK(vertices[0].Pos == v3f{0.0f, 0.0f, 0.0f});
		CHECK(vertices[1].Pos == v3f{1.0f, 0.0f, 0.0f});
		CHECK(vertices[2].Pos == v3f{0.0f, 1.0f, 0.0f});
	}

	SECTION("vertex indices are correct")
	{
		REQUIRE(sm.getMesh()->getMeshBuffer(0)->getIndexCount() == 3);
		const auto *indices = reinterpret_cast<irr::u16 *>(
				sm.getMesh()->getMeshBuffer(0)->getIndices());
		CHECK(indices[0] == 2);
		CHECK(indices[1] == 1);
		CHECK(indices[2] == 0);
	}
}

TEST_CASE("blender cube")
{
	ScopedMesh sm("source/Irrlicht/tests/assets/blender_cube.gltf");
	REQUIRE(sm.getMesh() != nullptr);
	REQUIRE(sm.getMesh()->getMeshBufferCount() == 1);
	SECTION("vertex coordinates are correct")
	{
		REQUIRE(sm.getMesh()->getMeshBuffer(0)->getVertexCount() == 24);
		const auto *vertices = reinterpret_cast<irr::video::S3DVertex *>(
				sm.getMesh()->getMeshBuffer(0)->getVertices());
		CHECK(vertices[0].Pos == v3f{-10.0f, -10.0f, -10.0f});
		CHECK(vertices[3].Pos == v3f{-10.0f, 10.0f, -10.0f});
		CHECK(vertices[6].Pos == v3f{-10.0f, -10.0f, 10.0f});
		CHECK(vertices[9].Pos == v3f{-10.0f, 10.0f, 10.0f});
		CHECK(vertices[12].Pos == v3f{10.0f, -10.0f, -10.0f});
		CHECK(vertices[15].Pos == v3f{10.0f, 10.0f, -10.0f});
		CHECK(vertices[18].Pos == v3f{10.0f, -10.0f, 10.0f});
		CHECK(vertices[21].Pos == v3f{10.0f, 10.0f, 10.0f});
	}

	SECTION("vertex indices are correct")
	{
		REQUIRE(sm.getMesh()->getMeshBuffer(0)->getIndexCount() == 36);
		const auto *indices = reinterpret_cast<irr::u16 *>(
				sm.getMesh()->getMeshBuffer(0)->getIndices());
		CHECK(indices[0] == 16);
		CHECK(indices[1] == 5);
		CHECK(indices[2] == 22);
		CHECK(indices[35] == 0);
	}

	SECTION("vertex normals are correct")
	{
		REQUIRE(sm.getMesh()->getMeshBuffer(0)->getVertexCount() == 24);
		const auto *vertices = reinterpret_cast<irr::video::S3DVertex *>(
				sm.getMesh()->getMeshBuffer(0)->getVertices());
		CHECK(vertices[0].Normal == v3f{-1.0f, 0.0f, 0.0f});
		CHECK(vertices[1].Normal == v3f{0.0f, -1.0f, 0.0f});
		CHECK(vertices[2].Normal == v3f{0.0f, 0.0f, -1.0f});
		CHECK(vertices[3].Normal == v3f{-1.0f, 0.0f, 0.0f});
		CHECK(vertices[6].Normal == v3f{-1.0f, 0.0f, 0.0f});
		CHECK(vertices[23].Normal == v3f{1.0f, 0.0f, 0.0f});
	}

	SECTION("texture coords are correct")
	{
		REQUIRE(sm.getMesh()->getMeshBuffer(0)->getVertexCount() == 24);
		const auto *vertices = reinterpret_cast<irr::video::S3DVertex *>(
				sm.getMesh()->getMeshBuffer(0)->getVertices());
		CHECK(vertices[0].TCoords == irr::core::vector2df{0.375f, 1.0f});
		CHECK(vertices[1].TCoords == irr::core::vector2df{0.125f, 0.25f});
		CHECK(vertices[2].TCoords == irr::core::vector2df{0.375f, 0.0f});
		CHECK(vertices[3].TCoords == irr::core::vector2df{0.6250f, 1.0f});
		CHECK(vertices[6].TCoords == irr::core::vector2df{0.375f, 0.75f});
	}
}

TEST_CASE("blender cube scaled")
{
	ScopedMesh sm("source/Irrlicht/tests/assets/blender_cube_scaled.gltf");
	REQUIRE(sm.getMesh() != nullptr);
	REQUIRE(sm.getMesh()->getMeshBufferCount() == 1);

	SECTION("Scaling is correct")
	{
		REQUIRE(sm.getMesh()->getMeshBuffer(0)->getVertexCount() == 24);
		const auto *vertices = reinterpret_cast<irr::video::S3DVertex *>(
				sm.getMesh()->getMeshBuffer(0)->getVertices());

		CHECK(vertices[0].Pos == v3f{-150.0f, -1.0f, -21.5f});
		CHECK(vertices[3].Pos == v3f{-150.0f, 1.0f, -21.5f});
		CHECK(vertices[6].Pos == v3f{-150.0f, -1.0f, 21.5f});
		CHECK(vertices[9].Pos == v3f{-150.0f, 1.0f, 21.5f});
		CHECK(vertices[12].Pos == v3f{150.0f, -1.0f, -21.5f});
		CHECK(vertices[15].Pos == v3f{150.0f, 1.0f, -21.5f});
		CHECK(vertices[18].Pos == v3f{150.0f, -1.0f, 21.5f});
		CHECK(vertices[21].Pos == v3f{150.0f, 1.0f, 21.5f});
	}
}

TEST_CASE("snow man")
{
	ScopedMesh sm("source/Irrlicht/tests/assets/snow_man.gltf");
	REQUIRE(sm.getMesh() != nullptr);
	REQUIRE(sm.getMesh()->getMeshBufferCount() == 3);

	SECTION("vertex coordinates are correct for all buffers")
	{
		REQUIRE(sm.getMesh()->getMeshBuffer(0)->getVertexCount() == 24);
		const auto *vertices = reinterpret_cast<irr::video::S3DVertex *>(
				sm.getMesh()->getMeshBuffer(0)->getVertices());

		CHECK(vertices[0].Pos == v3f{3.0f, 24.0f, -3.0f});
		CHECK(vertices[3].Pos == v3f{3.0f, 18.0f, 3.0f});
		CHECK(vertices[6].Pos == v3f{-3.0f, 18.0f, -3.0f});
		CHECK(vertices[9].Pos == v3f{3.0f, 24.0f, 3.0f});
		CHECK(vertices[12].Pos == v3f{3.0f, 18.0f, -3.0f});
		CHECK(vertices[15].Pos == v3f{-3.0f, 18.0f, 3.0f});
		CHECK(vertices[18].Pos == v3f{3.0f, 18.0f, -3.0f});
		CHECK(vertices[21].Pos == v3f{3.0f, 18.0f, 3.0f});

		vertices = reinterpret_cast<irr::video::S3DVertex *>(
				sm.getMesh()->getMeshBuffer(1)->getVertices());

		CHECK(vertices[2].Pos == v3f{5.0f, 10.0f, 5.0f});
		CHECK(vertices[3].Pos == v3f{5.0f, 0.0f, 5.0f});
		CHECK(vertices[7].Pos == v3f{-5.0f, 0.0f, 5.0f});
		CHECK(vertices[8].Pos == v3f{5.0f, 10.0f, -5.0f});
		CHECK(vertices[14].Pos == v3f{5.0f, 0.0f, 5.0f});
		CHECK(vertices[16].Pos == v3f{5.0f, 10.0f, -5.0f});
		CHECK(vertices[22].Pos == v3f{-5.0f, 10.0f, 5.0f});
		CHECK(vertices[23].Pos == v3f{-5.0f, 0.0f, 5.0f});

		vertices = reinterpret_cast<irr::video::S3DVertex *>(
				sm.getMesh()->getMeshBuffer(2)->getVertices());

		CHECK(vertices[1].Pos == v3f{4.0f, 10.0f, -4.0f});
		CHECK(vertices[2].Pos == v3f{4.0f, 18.0f, 4.0f});
		CHECK(vertices[3].Pos == v3f{4.0f, 10.0f, 4.0f});
		CHECK(vertices[10].Pos == v3f{-4.0f, 18.0f, -4.0f});
		CHECK(vertices[11].Pos == v3f{-4.0f, 18.0f, 4.0f});
		CHECK(vertices[12].Pos == v3f{4.0f, 10.0f, -4.0f});
		CHECK(vertices[17].Pos == v3f{-4.0f, 18.0f, -4.0f});
		CHECK(vertices[18].Pos == v3f{4.0f, 10.0f, -4.0f});
	}

	SECTION("vertex indices are correct for all buffers")
	{
		REQUIRE(sm.getMesh()->getMeshBuffer(0)->getIndexCount() == 36);
		const auto *indices = reinterpret_cast<irr::u16 *>(
				sm.getMesh()->getMeshBuffer(0)->getIndices());
		CHECK(indices[0] == 23);
		CHECK(indices[1] == 21);
		CHECK(indices[2] == 22);
		CHECK(indices[35] == 2);

		REQUIRE(sm.getMesh()->getMeshBuffer(1)->getIndexCount() == 36);
		indices = reinterpret_cast<irr::u16 *>(
				sm.getMesh()->getMeshBuffer(1)->getIndices());
		CHECK(indices[10] == 16);
		CHECK(indices[11] == 18);
		CHECK(indices[15] == 13);
		CHECK(indices[27] == 5);

		REQUIRE(sm.getMesh()->getMeshBuffer(1)->getIndexCount() == 36);
		indices = reinterpret_cast<irr::u16 *>(
				sm.getMesh()->getMeshBuffer(2)->getIndices());
		CHECK(indices[26] == 6);
		CHECK(indices[27] == 5);
		CHECK(indices[29] == 6);
		CHECK(indices[32] == 2);
	}

	SECTION("vertex normals are correct for all buffers")
	{
		REQUIRE(sm.getMesh()->getMeshBuffer(0)->getVertexCount() == 24);
		const auto *vertices = reinterpret_cast<irr::video::S3DVertex *>(
				sm.getMesh()->getMeshBuffer(0)->getVertices());
		CHECK(vertices[0].Normal == v3f{1.0f, 0.0f, -0.0f});
		CHECK(vertices[1].Normal == v3f{1.0f, 0.0f, -0.0f});
		CHECK(vertices[2].Normal == v3f{1.0f, 0.0f, -0.0f});
		CHECK(vertices[3].Normal == v3f{1.0f, 0.0f, -0.0f});
		CHECK(vertices[6].Normal == v3f{-1.0f, 0.0f, -0.0f});
		CHECK(vertices[23].Normal == v3f{0.0f, 0.0f, 1.0f});

		vertices = reinterpret_cast<irr::video::S3DVertex *>(
				sm.getMesh()->getMeshBuffer(1)->getVertices());

		CHECK(vertices[0].Normal == v3f{1.0f, 0.0f, -0.0f});
		CHECK(vertices[1].Normal == v3f{1.0f, 0.0f, -0.0f});
		CHECK(vertices[3].Normal == v3f{1.0f, 0.0f, -0.0f});
		CHECK(vertices[6].Normal == v3f{-1.0f, 0.0f, -0.0f});
		CHECK(vertices[7].Normal == v3f{-1.0f, 0.0f, -0.0f});
		CHECK(vertices[22].Normal == v3f{0.0f, 0.0f, 1.0f});

		vertices = reinterpret_cast<irr::video::S3DVertex *>(
				sm.getMesh()->getMeshBuffer(2)->getVertices());

		CHECK(vertices[3].Normal == v3f{1.0f, 0.0f, -0.0f});
		CHECK(vertices[4].Normal == v3f{-1.0f, 0.0f, -0.0f});
		CHECK(vertices[5].Normal == v3f{-1.0f, 0.0f, -0.0f});
		CHECK(vertices[10].Normal == v3f{0.0f, 1.0f, -0.0f});
		CHECK(vertices[11].Normal == v3f{0.0f, 1.0f, -0.0f});
		CHECK(vertices[19].Normal == v3f{0.0f, 0.0f, -1.0f});
	}

	SECTION("texture coords are correct for all buffers")
	{
		REQUIRE(sm.getMesh()->getMeshBuffer(0)->getVertexCount() == 24);
		const auto *vertices = reinterpret_cast<irr::video::S3DVertex *>(
				sm.getMesh()->getMeshBuffer(0)->getVertices());

		CHECK(vertices[0].TCoords == irr::core::vector2df{0.583333, 0.791667});
		CHECK(vertices[1].TCoords == irr::core::vector2df{0.583333, 0.666667});
		CHECK(vertices[2].TCoords == irr::core::vector2df{0.708333, 0.791667});
		CHECK(vertices[5].TCoords == irr::core::vector2df{0.375, 0.416667});
		CHECK(vertices[6].TCoords == irr::core::vector2df{0.5, 0.291667});
		CHECK(vertices[19].TCoords == irr::core::vector2df{0.708333, 0.75});

		vertices = reinterpret_cast<irr::video::S3DVertex *>(
				sm.getMesh()->getMeshBuffer(1)->getVertices());

		CHECK(vertices[1].TCoords == irr::core::vector2df{0, 0.791667});
		CHECK(vertices[4].TCoords == irr::core::vector2df{0.208333, 0.791667});
		CHECK(vertices[5].TCoords == irr::core::vector2df{0, 0.791667});
		CHECK(vertices[6].TCoords == irr::core::vector2df{0.208333, 0.583333});
		CHECK(vertices[12].TCoords == irr::core::vector2df{0.416667, 0.791667});
		CHECK(vertices[15].TCoords == irr::core::vector2df{0.208333, 0.583333});

		vertices = reinterpret_cast<irr::video::S3DVertex *>(
				sm.getMesh()->getMeshBuffer(2)->getVertices());

		CHECK(vertices[10].TCoords == irr::core::vector2df{0.375, 0.416667});
		CHECK(vertices[11].TCoords == irr::core::vector2df{0.375, 0.583333});
		CHECK(vertices[12].TCoords == irr::core::vector2df{0.708333, 0.625});
		CHECK(vertices[17].TCoords == irr::core::vector2df{0.541667, 0.458333});
		CHECK(vertices[20].TCoords == irr::core::vector2df{0.208333, 0.416667});
		CHECK(vertices[22].TCoords == irr::core::vector2df{0.375, 0.416667});
	}
}

// https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/SimpleSparseAccessor
TEST_CASE("simple sparse accessor")
{
	ScopedMesh sm("source/Irrlicht/tests/assets/simple_sparse_accessor.gltf");
	REQUIRE(sm.getMesh() != nullptr);
	const auto *vertices = reinterpret_cast<irr::video::S3DVertex *>(
			sm.getMesh()->getMeshBuffer(0)->getVertices());
	const std::array<v3f, 14> expectedPositions = {
			// Lower
			v3f(0, 0, 0),
			v3f(1, 0, 0),
			v3f(2, 0, 0),
			v3f(3, 0, 0),
			v3f(4, 0, 0),
			v3f(5, 0, 0),
			v3f(6, 0, 0),
			// Upper
			v3f(0, 1, 0),
			v3f(1, 2, 0), // overridden
			v3f(2, 1, 0),
			v3f(3, 3, 0), // overridden
			v3f(4, 1, 0),
			v3f(5, 4, 0), // overridden
			v3f(6, 1, 0),
	};
	for (std::size_t i = 0; i < expectedPositions.size(); ++i) {
		CHECK(vertices[i].Pos == expectedPositions[i]);
	}
}

// https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/SimpleSkin
TEST_CASE("simple skin")
{
	using CSkinnedMesh = irr::scene::CSkinnedMesh;
	ScopedMesh sm("source/Irrlicht/tests/assets/simple_skin.gltf");
	REQUIRE(sm.getMesh() != nullptr);
	auto csm = dynamic_cast<const CSkinnedMesh*>(sm.getMesh());
	const auto joints = csm->getAllJoints();
	REQUIRE(joints.size() == 3);

	const auto findJoint = [&](const std::function<bool(CSkinnedMesh::SJoint*)> &predicate) {
		for (std::size_t i = 0; i < joints.size(); ++i) {
			if (predicate(joints[i])) {
				return joints[i];
			}
		}
		throw std::runtime_error("joint not found");
	};

	// Check the node hierarchy
	const auto parent = findJoint([](auto joint) {
		return !joint->Children.empty();
	});
	REQUIRE(parent->Children.size() == 1);
	const auto child = parent->Children[0];
	REQUIRE(child != parent);

	SECTION("transformations are correct")
	{
		CHECK(parent->Animatedposition == v3f(0, 0, 0));
		CHECK(parent->Animatedrotation == irr::core::quaternion());
		CHECK(parent->Animatedscale == v3f(1, 1, 1));
		CHECK(parent->GlobalInversedMatrix == irr::core::matrix4());
		const v3f childTranslation(0, 1, 0);
		CHECK(child->Animatedposition == childTranslation);
		CHECK(child->Animatedrotation == irr::core::quaternion());
		CHECK(child->Animatedscale == v3f(1, 1, 1));
		irr::core::matrix4 inverseBindMatrix;
		inverseBindMatrix.setInverseTranslation(childTranslation);
		CHECK(child->GlobalInversedMatrix == inverseBindMatrix);
	}

	SECTION("weights are correct")
	{
		const auto weights = [&](const CSkinnedMesh::SJoint* joint) {
			std::unordered_map<irr::u32, irr::f32> weights;
			for (std::size_t i = 0; i < joint->Weights.size(); ++i) {
				const auto weight = joint->Weights[i];
				REQUIRE(weight.buffer_id == 0);
				weights[weight.vertex_id] = weight.strength;
			}
			return weights;
		};
		const auto parentWeights = weights(parent);
		const auto childWeights = weights(child);
		
		const auto checkWeights = [&](irr::u32 index, irr::f32 parentWeight, irr::f32 childWeight) {
			const auto getWeight = [](auto weights, auto index) {
				const auto it = weights.find(index);
				return it == weights.end() ? 0.0f : it->second;
			};
			CHECK(getWeight(parentWeights, index) == parentWeight);
			CHECK(getWeight(childWeights, index) == childWeight);
		};
		checkWeights(0, 1.00, 0.00);
		checkWeights(1, 1.00, 0.00);
		checkWeights(2, 0.75, 0.25);
		checkWeights(3, 0.75, 0.25);
		checkWeights(4, 0.50, 0.50);
		checkWeights(5, 0.50, 0.50);
		checkWeights(6, 0.25, 0.75);
		checkWeights(7, 0.25, 0.75);
		checkWeights(8, 0.00, 1.00);
		checkWeights(9, 0.00, 1.00);
	}

	SECTION("there should be a third node not involved in skinning")
	{
		const auto other = findJoint([&](auto joint) {
			return joint != child && joint != parent;
		});
		CHECK(other->Weights.empty());
	}
}
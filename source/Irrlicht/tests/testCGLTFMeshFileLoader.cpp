#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <irrlicht.h>

class ScopedMesh
{
public:
	ScopedMesh(irr::io::IReadFile* file)
		: m_device { irr::createDevice(irr::video::EDT_NULL) }
		, m_mesh { nullptr }
	{
		auto* smgr = m_device->getSceneManager();
		m_mesh = smgr->getMesh(file);
	}

	ScopedMesh(const irr::io::path& filepath)
		: m_device { irr::createDevice(irr::video::EDT_NULL) }
		, m_mesh { nullptr }
	{
		auto* smgr = m_device->getSceneManager();
		m_mesh = smgr->getMesh(filepath, "");
	}

	~ScopedMesh()
	{
		m_device->drop();
		m_mesh = nullptr;
	}

	const irr::scene::IAnimatedMesh* getMesh() const
	{
		return m_mesh;
	}

private:
	irr::IrrlichtDevice* m_device;
	irr::scene::IAnimatedMesh* m_mesh;
};

TEST_CASE("load empty gltf file") {
	ScopedMesh sm("source/Irrlicht/tests/assets/empty.gltf");
	CHECK(sm.getMesh() == nullptr);
}

TEST_CASE("minimal triangle") {
	ScopedMesh sm("source/Irrlicht/tests/assets/minimal_triangle.gltf");
	REQUIRE(sm.getMesh() != nullptr);
	REQUIRE(sm.getMesh()->getMeshBufferCount() == 1);

	SECTION("vertex coordinates are correct") {
		REQUIRE(sm.getMesh()->getMeshBuffer(0)->getVertexCount() == 3);
		const auto* vertices = reinterpret_cast<irr::video::S3DVertex*>(
			sm.getMesh()->getMeshBuffer(0)->getVertices());
		CHECK(vertices[0].Pos == irr::core::vector3df {0.0f, 0.0f, 0.0f});
		CHECK(vertices[1].Pos == irr::core::vector3df {1.0f, 0.0f, 0.0f});
		CHECK(vertices[2].Pos == irr::core::vector3df {0.0f, 1.0f, 0.0f});
	}

	SECTION("vertex indices are correct") {
		REQUIRE(sm.getMesh()->getMeshBuffer(0)->getIndexCount() == 3);
		const auto* indices = reinterpret_cast<irr::u16*>(
			sm.getMesh()->getMeshBuffer(0)->getIndices());
		CHECK(indices[0] == 0);
		CHECK(indices[1] == 1);
		CHECK(indices[2] == 2);
	}
}

TEST_CASE("blender cube") {
	ScopedMesh sm("source/Irrlicht/tests/assets/blender_cube.gltf");
	REQUIRE(sm.getMesh() != nullptr);
	REQUIRE(sm.getMesh()->getMeshBufferCount() == 1);
	SECTION("vertex coordinates are correct") {
		REQUIRE(sm.getMesh()->getMeshBuffer(0)->getVertexCount() == 24);
		const auto* vertices = reinterpret_cast<irr::video::S3DVertex*>(
			sm.getMesh()->getMeshBuffer(0)->getVertices());
		CHECK(vertices[0].Pos == irr::core::vector3df{-10.0f, -10.0f, -10.0f});
		CHECK(vertices[3].Pos == irr::core::vector3df{-10.0f, 10.0f, -10.0f});
		CHECK(vertices[6].Pos == irr::core::vector3df{-10.0f, -10.0f, 10.0f});
		CHECK(vertices[9].Pos == irr::core::vector3df{-10.0f, 10.0f, 10.0f});
		CHECK(vertices[12].Pos == irr::core::vector3df{10.0f, -10.0f, -10.0f});
		CHECK(vertices[15].Pos == irr::core::vector3df{10.0f, 10.0f, -10.0f});
		CHECK(vertices[18].Pos == irr::core::vector3df{10.0f, -10.0f, 10.0f});
		CHECK(vertices[21].Pos == irr::core::vector3df{10.0f, 10.0f, 10.0f});
	}

	SECTION("vertex indices are correct") {
		REQUIRE(sm.getMesh()->getMeshBuffer(0)->getIndexCount() == 36);
		const auto* indices = reinterpret_cast<irr::u16*>(
			sm.getMesh()->getMeshBuffer(0)->getIndices());
		CHECK(indices[0] == 0);
		CHECK(indices[1] == 3);
		CHECK(indices[2] == 9);
	}

	SECTION("vertex normals are correct") {
		REQUIRE(sm.getMesh()->getMeshBuffer(0)->getVertexCount() == 24);
		const auto* vertices = reinterpret_cast<irr::video::S3DVertex*>(
			sm.getMesh()->getMeshBuffer(0)->getVertices());
		CHECK(vertices[0].Normal == irr::core::vector3df{-1.0f, 0.0f, 0.0f});
		CHECK(vertices[1].Normal == irr::core::vector3df{0.0f, -1.0f, 0.0f});
		CHECK(vertices[2].Normal == irr::core::vector3df{0.0f, 0.0f, -1.0f});
		CHECK(vertices[3].Normal == irr::core::vector3df{-1.0f, 0.0f, 0.0f});
		CHECK(vertices[6].Normal == irr::core::vector3df{-1.0f, 0.0f, 0.0f});
		CHECK(vertices[23].Normal == irr::core::vector3df{1.0f, 0.0f, 0.0f});

	}

	SECTION("texture coords are correct") {
		REQUIRE(sm.getMesh()->getMeshBuffer(0)->getVertexCount() == 24);
		const auto* vertices = reinterpret_cast<irr::video::S3DVertex*>(
			sm.getMesh()->getMeshBuffer(0)->getVertices());
		CHECK(vertices[0].TCoords == irr::core::vector2df{0.375f, 1.0f});
		CHECK(vertices[1].TCoords == irr::core::vector2df{0.125f, 0.25f});
		CHECK(vertices[2].TCoords == irr::core::vector2df{0.375f, 0.0f});
		CHECK(vertices[3].TCoords == irr::core::vector2df{0.6250f, 1.0f});
		CHECK(vertices[6].TCoords == irr::core::vector2df{0.375f, 0.75f});
	}
}

TEST_CASE("mesh loader returns nullptr when given null file pointer") {
	ScopedMesh sm(nullptr);
	CHECK(sm.getMesh() == nullptr);
}

TEST_CASE("invalid JSON returns nullptr") {
	ScopedMesh sm("source/Irrlicht/tests/assets/json_missing_brace.gltf");
	CHECK(sm.getMesh() == nullptr);
}


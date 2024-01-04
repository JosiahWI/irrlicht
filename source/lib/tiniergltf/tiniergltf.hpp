#pragma once

#include <functional>
#include <stack>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
#include <array>
#include <optional>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <json/json.h>
#include "base64.h"

namespace tiniergltf {

#define CHECK(cond) do { \
		if (!(cond)) \
			throw std::runtime_error(__FILE__ ":" \
				+ std::to_string(__LINE__) + \
				": check failed"); \
	} while (0)

template <typename T>
static inline void checkIndex(std::optional<std::vector<T>> vec, std::optional<std::size_t> i) {
	if (!i.has_value()) return;
	CHECK(vec.has_value());
	CHECK(i < vec->size());
}

template <typename T>
static inline void checkIndex(std::vector<T> vec, std::optional<std::size_t> i) {
	if (!i.has_value()) return;
	CHECK(i < vec.size());
}

template <typename T, typename F>
static inline void checkForall(std::optional<std::vector<T>> vec, F cond) {
	if (!vec.has_value())
		return;
	for (const T &v : vec.value())
		cond(v);
}

template <typename T>
static inline T as(const Json::Value &o);

template<>
bool as(const Json::Value &o) {
	CHECK(o.isBool());
	return o.asBool();
}

template<>
double as (const Json::Value &o) {
	CHECK(o.isDouble());
	return o.asDouble();
}

template<>
std::size_t as(const Json::Value &o) {
	CHECK(o.isUInt64());
	auto u = o.asUInt64();
	CHECK(u <= std::numeric_limits<std::size_t>::max());
	return u;
}

template<>
std::string as(const Json::Value &o) {
	CHECK(o.isString());
	return o.asString();
}

template<typename U>
std::vector<U> asVec(const Json::Value &o) {
	CHECK(o.isArray());
	std::vector<U> res;
	res.reserve(o.size());
	for (Json::ArrayIndex i = 0; i < o.size(); ++i) {
		res.push_back(as<U>(o[i]));
	}
	return res;
}

template<typename U, std::size_t n>
std::array<U, n> asArr(const Json::Value &o) {
	CHECK(o.isArray());
	CHECK(o.size() == n);
	std::array<U, n> res;
	for (Json::ArrayIndex i = 0; i < n; ++i) {
		res[i] = as<U>(o[i]);
	}
	return res;
}

struct AccessorSparseIndices {
	std::size_t bufferView;
	std::size_t byteOffset;
	enum class ComponentType {
		UNSIGNED_BYTE,
		UNSIGNED_SHORT,
		UNSIGNED_INT,
	};
	static std::size_t componentSize(ComponentType type) {
		switch (type) {
		case ComponentType::UNSIGNED_BYTE:
			return 1;
		case ComponentType::UNSIGNED_SHORT:
			return 2;
		case ComponentType::UNSIGNED_INT:
			return 4;
		}
		throw std::logic_error("invalid component type");
	}
	ComponentType componentType;
	std::size_t elementSize() const {
		return componentSize(componentType);
	}
	AccessorSparseIndices(const Json::Value &o)
		: bufferView(as<std::size_t>(o["bufferView"]))
		, byteOffset(0)
	{
		CHECK(o.isObject());
		if (o.isMember("byteOffset")) {
			byteOffset = as<std::size_t>(o["byteOffset"]);
			CHECK(byteOffset >= 0);
		}
		{
			static std::unordered_map<Json::UInt64, ComponentType> map = {
				{5121, ComponentType::UNSIGNED_BYTE},
				{5123, ComponentType::UNSIGNED_SHORT},
				{5125, ComponentType::UNSIGNED_INT},
			};
			auto v = o["componentType"]; CHECK(v.isUInt64());
			componentType = map.at(v.asUInt64());
		}
	}
};
template<> AccessorSparseIndices as(const Json::Value &o) { return o; }

struct AccessorSparseValues {
	std::size_t bufferView;
	std::size_t byteOffset;
	AccessorSparseValues(const Json::Value &o)
		: bufferView(as<std::size_t>(o["bufferView"]))
		, byteOffset(0)
	{
		CHECK(o.isObject());
		if (o.isMember("byteOffset")) {
			byteOffset = as<std::size_t>(o["byteOffset"]);
			CHECK(byteOffset >= 0);
		}
	}
};
template<> AccessorSparseValues as(const Json::Value &o) { return o; }

struct AccessorSparse {
	std::size_t count;
	AccessorSparseIndices indices;
	AccessorSparseValues values;
	AccessorSparse(const Json::Value &o)
		: count(as<std::size_t>(o["count"]))
		, indices(as<AccessorSparseIndices>(o["indices"]))
		, values(as<AccessorSparseValues>(o["values"]))
	{
		CHECK(o.isObject());
		CHECK(count >= 1);
	}
};
template<> AccessorSparse as(const Json::Value &o) { return o; }

struct Accessor {
	std::optional<std::size_t> bufferView;
	std::size_t byteOffset;
	enum class ComponentType {
		BYTE,
		UNSIGNED_BYTE,
		SHORT,
		UNSIGNED_SHORT,
		UNSIGNED_INT,
		FLOAT,
	};
	ComponentType componentType;
	static std::size_t componentSize(ComponentType type) {
		switch (type) {
		case ComponentType::BYTE:
		case ComponentType::UNSIGNED_BYTE:
			return 1;
		case ComponentType::SHORT:
		case ComponentType::UNSIGNED_SHORT:
			return 2;
		case ComponentType::UNSIGNED_INT:
		case ComponentType::FLOAT:
			return 4;
		}
		throw std::logic_error("invalid component type");
	}
	std::size_t count;
	std::optional<std::vector<double>> max;
	std::optional<std::vector<double>> min;
	std::optional<std::string> name;
	bool normalized;
	std::optional<AccessorSparse> sparse;
	enum class Type {
		MAT2,
		MAT3,
		MAT4,
		SCALAR,
		VEC2,
		VEC3,
		VEC4,
	};
	static std::size_t typeCount(Type type) {
		switch (type) {
		case Type::SCALAR:
			return 1;
		case Type::VEC2:
			return 2;
		case Type::VEC3:
			return 3;
		case Type::MAT2:
		case Type::VEC4:
			return 4;
		case Type::MAT3:
			return 9;
		case Type::MAT4:
			return 16;
		}
		throw std::logic_error("invalid type");
	}
	Type type;
	std::size_t elementSize() const {
		return componentSize(componentType) * typeCount(type);
	}
	Accessor(const Json::Value &o)
		: byteOffset(0)
		, count(as<std::size_t>(o["count"]))
		, normalized(false)
	{
		CHECK(o.isObject());
		if (o.isMember("bufferView")) {
			bufferView = as<std::size_t>(o["bufferView"]);
		}
		{
			static std::unordered_map<Json::UInt64, ComponentType> map = {
				{5120, ComponentType::BYTE},
				{5121, ComponentType::UNSIGNED_BYTE},
				{5122, ComponentType::SHORT},
				{5123, ComponentType::UNSIGNED_SHORT},
				{5125, ComponentType::UNSIGNED_INT},
				{5126, ComponentType::FLOAT},
			};
			auto v = o["componentType"]; CHECK(v.isUInt64());
			componentType = map.at(v.asUInt64());
		}
		if (o.isMember("byteOffset")) {
			byteOffset = as<std::size_t>(o["byteOffset"]);
			CHECK(byteOffset >= 0);
			CHECK(byteOffset % componentSize(componentType) == 0);
		}
		CHECK(count >= 1);
		if (o.isMember("name")) {
			name = as<std::string>(o["name"]);
		}
		if (o.isMember("normalized")) {
			normalized = as<bool>(o["normalized"]);
		}
		if (o.isMember("sparse")) {
			sparse = as<AccessorSparse>(o["sparse"]);
		}
		{
			static std::unordered_map<Json::String, Type> map = {
				{"MAT2", Type::MAT2},
				{"MAT3", Type::MAT3},
				{"MAT4", Type::MAT4},
				{"SCALAR", Type::SCALAR},
				{"VEC2", Type::VEC2},
				{"VEC3", Type::VEC3},
				{"VEC4", Type::VEC4},
			};
			auto v = o["type"]; CHECK(v.isString());
			type = map.at(v.asString());
		}
		if (o.isMember("max")) {
			max = asVec<double>(o["max"]);
			CHECK(max->size() == typeCount(type));
		}
		if (o.isMember("min")) {
			min = asVec<double>(o["min"]);
			CHECK(min->size() == typeCount(type));
		}
	}
};
template<> Accessor as(const Json::Value &o) { return o; }

struct AnimationChannelTarget {
	std::optional<std::size_t> node;
	enum class Path {
		ROTATION,
		SCALE,
		TRANSLATION,
		WEIGHTS,
	};
	Path path;
	AnimationChannelTarget(const Json::Value &o)
	{
		CHECK(o.isObject());
		if (o.isMember("node")) {
			node = as<std::size_t>(o["node"]);
		}
		{
			static std::unordered_map<Json::String, Path> map = {
				{"rotation", Path::ROTATION},
				{"scale", Path::SCALE},
				{"translation", Path::TRANSLATION},
				{"weights", Path::WEIGHTS},
			};
			auto v = o["path"]; CHECK(v.isString());
			path = map.at(v.asString());
		}
	}
};
template<> AnimationChannelTarget as(const Json::Value &o) { return o; }

struct AnimationChannel {
	std::size_t sampler;
	AnimationChannelTarget target;
	AnimationChannel(const Json::Value &o)
		: sampler(as<std::size_t>(o["sampler"]))
		, target(as<AnimationChannelTarget>(o["target"]))
	{
		CHECK(o.isObject());
	}
};
template<> AnimationChannel as(const Json::Value &o) { return o; }

struct AnimationSampler {
	std::size_t input;
	enum class Interpolation {
		CUBICSPLINE,
		LINEAR,
		STEP,
	};
	Interpolation interpolation;
	std::size_t output;
	AnimationSampler(const Json::Value &o)
		: input(as<std::size_t>(o["input"]))
		, interpolation(Interpolation::LINEAR)
		, output(as<std::size_t>(o["output"]))
	{
		CHECK(o.isObject());
		if (o.isMember("interpolation")) {
			static std::unordered_map<Json::String, Interpolation> map = {
				{"CUBICSPLINE", Interpolation::CUBICSPLINE},
				{"LINEAR", Interpolation::LINEAR},
				{"STEP", Interpolation::STEP},
			};
			auto v = o["interpolation"]; CHECK(v.isString());
			interpolation = map.at(v.asString());
		}
	}
};
template<> AnimationSampler as(const Json::Value &o) { return o; }

struct Animation {
	std::vector<AnimationChannel> channels;
	std::optional<std::string> name;
	std::vector<AnimationSampler> samplers;
	Animation(const Json::Value &o)
		: channels(asVec<AnimationChannel>(o["channels"]))
		, samplers(asVec<AnimationSampler>(o["samplers"]))
	{
		CHECK(o.isObject());
		CHECK(channels.size() >= 1);
		if (o.isMember("name")) {
			name = as<std::string>(o["name"]);
		}
		CHECK(samplers.size() >= 1);
	}
};
template<> Animation as(const Json::Value &o) { return o; }

struct Asset {
	std::optional<std::string> copyright;
	std::optional<std::string> generator;
	std::optional<std::string> minVersion;
	std::string version;
	Asset(const Json::Value &o)
		: version(as<std::string>(o["version"]))
	{
		CHECK(o.isObject());
		if (o.isMember("copyright")) {
			copyright = as<std::string>(o["copyright"]);
		}
		if (o.isMember("generator")) {
			generator = as<std::string>(o["generator"]);
		}
		if (o.isMember("minVersion")) {
			minVersion = as<std::string>(o["minVersion"]);
		}
	}
};
template<> Asset as(const Json::Value &o) { return o; }

struct BufferView {
	std::size_t buffer;
	std::size_t byteLength;
	std::size_t byteOffset;
	std::optional<std::size_t> byteStride;
	std::optional<std::string> name;
	enum class Target {
		ARRAY_BUFFER,
		ELEMENT_ARRAY_BUFFER,
	};
	std::optional<Target> target;
	BufferView(const Json::Value &o)
		: buffer(as<std::size_t>(o["buffer"]))
		, byteLength(as<std::size_t>(o["byteLength"]))
		, byteOffset(0)
	{
		CHECK(o.isObject());
		CHECK(byteLength >= 1);
		if (o.isMember("byteOffset")) {
			byteOffset = as<std::size_t>(o["byteOffset"]);
			CHECK(byteOffset >= 0);
		}
		if (o.isMember("byteStride")) {
			byteStride = as<std::size_t>(o["byteStride"]);
			CHECK(byteStride.value() >= 4);
			CHECK(byteStride.value() <= 252);
			CHECK(byteStride.value() % 4 == 0);
		}
		if (o.isMember("name")) {
			name = as<std::string>(o["name"]);
		}
		if (o.isMember("target")) {
			static std::unordered_map<Json::UInt64, Target> map = {
				{34962, Target::ARRAY_BUFFER},
				{34963, Target::ELEMENT_ARRAY_BUFFER},
			};
			auto v = o["target"]; CHECK(v.isUInt64());
			target = map.at(v.asUInt64());
		}
	}
};
template<> BufferView as(const Json::Value &o) { return o; }

struct Buffer {
	std::size_t byteLength;
	std::optional<std::string> name;
	std::vector<unsigned char> data;
	Buffer(const Json::Value &o,
			const std::function<std::vector<unsigned char>(const std::string &uri)> &resolveURI)
		: byteLength(as<std::size_t>(o["byteLength"]))
	{
		CHECK(o.isObject());
		CHECK(byteLength >= 1);
		if (o.isMember("name")) {
			name = as<std::string>(o["name"]);
		}
		CHECK(o.isMember("uri"));
		bool dataURI = false;
		const std::string uri = as<std::string>(o["uri"]);
		for (auto &prefix : std::array<std::string, 2> {
			"data:application/octet-stream;base64,",
			"data:application/gltf-buffer;base64,"
		}) {
			if (std::string_view(uri).substr(0, prefix.length()) == prefix) {
				auto view = std::string_view(uri).substr(prefix.length());
				CHECK(base64_is_valid(view));
				data = base64_decode(view);
				dataURI = true;
				break;
			}
		}
		if (!dataURI)
			data = resolveURI(uri);
		CHECK(data.size() >= byteLength);
		data.resize(byteLength);
	}
};

struct CameraOrthographic {
	double xmag;
	double ymag;
	double zfar;
	double znear;
	CameraOrthographic(const Json::Value &o)
		: xmag(as<double>(o["xmag"]))
		, ymag(as<double>(o["ymag"]))
		, zfar(as<double>(o["zfar"]))
		, znear(as<double>(o["znear"]))
	{
		CHECK(o.isObject());
		CHECK(zfar > 0);
		CHECK(znear >= 0);
	}
};
template<> CameraOrthographic as(const Json::Value &o) { return o; }

struct CameraPerspective {
	std::optional<double> aspectRatio;
	double yfov;
	std::optional<double> zfar;
	double znear;
	CameraPerspective(const Json::Value &o)
		: yfov(as<double>(o["yfov"]))
		, znear(as<double>(o["znear"]))
	{
		CHECK(o.isObject());
		if (o.isMember("aspectRatio")) {
			aspectRatio = as<double>(o["aspectRatio"]);
			CHECK(aspectRatio.value() > 0);
		}
		CHECK(yfov > 0);
		if (o.isMember("zfar")) {
			zfar = as<double>(o["zfar"]);
			CHECK(zfar.value() > 0);
		}
		CHECK(znear > 0);
	}
};
template<> CameraPerspective as(const Json::Value &o) { return o; }

struct Camera {
	std::optional<std::string> name;
	std::optional<CameraOrthographic> orthographic;
	std::optional<CameraPerspective> perspective;
	enum class Type {
		ORTHOGRAPHIC,
		PERSPECTIVE,
	};
	Type type;
	Camera(const Json::Value &o)
	{
		CHECK(o.isObject());
		if (o.isMember("name")) {
			name = as<std::string>(o["name"]);
		}
		if (o.isMember("orthographic")) {
			orthographic = as<CameraOrthographic>(o["orthographic"]);
		}
		if (o.isMember("perspective")) {
			perspective = as<CameraPerspective>(o["perspective"]);
		}
		{
			static std::unordered_map<Json::String, Type> map = {
				{"orthographic", Type::ORTHOGRAPHIC},
				{"perspective", Type::PERSPECTIVE},
			};
			auto v = o["type"]; CHECK(v.isString());
			type = map.at(v.asString());
		}
	}
};
template<> Camera as(const Json::Value &o) { return o; }

struct Image {
	std::optional<std::size_t> bufferView;
	enum class MimeType {
		IMAGE_JPEG,
		IMAGE_PNG,
	};
	std::optional<MimeType> mimeType;
	std::optional<std::string> name;
	std::optional<std::string> uri;
	Image(const Json::Value &o)
	{
		CHECK(o.isObject());
		if (o.isMember("bufferView")) {
			bufferView = as<std::size_t>(o["bufferView"]);
		}
		if (o.isMember("mimeType")) {
			static std::unordered_map<Json::String, MimeType> map = {
				{"image/jpeg", MimeType::IMAGE_JPEG},
				{"image/png", MimeType::IMAGE_PNG},
			};
			auto v = o["mimeType"]; CHECK(v.isString());
			mimeType = map.at(v.asString());
		}
		if (o.isMember("name")) {
			name = as<std::string>(o["name"]);
		}
		if (o.isMember("uri")) {
			uri = as<std::string>(o["uri"]);
		}
	}
};
template<> Image as(const Json::Value &o) { return o; }

struct TextureInfo {
	std::size_t index;
	std::size_t texCoord;
	TextureInfo(const Json::Value &o)
		: index(as<std::size_t>(o["index"]))
		, texCoord(0)
	{
		CHECK(o.isObject());
		if (o.isMember("texCoord")) {
			texCoord = as<std::size_t>(o["texCoord"]);
			CHECK(texCoord >= 0);
		}
	}
};
template<> TextureInfo as(const Json::Value &o) { return o; }

struct MaterialNormalTextureInfo {
	std::size_t index;
	double scale;
	std::size_t texCoord;
	MaterialNormalTextureInfo(const Json::Value &o)
		: index(as<std::size_t>(o["index"]))
		, scale(1)
		, texCoord(0)
	{
		CHECK(o.isObject());
		if (o.isMember("scale")) {
			scale = as<double>(o["scale"]);
		}
		if (o.isMember("texCoord")) {
			texCoord = as<std::size_t>(o["texCoord"]);
		}
	}
};
template<> MaterialNormalTextureInfo as(const Json::Value &o) { return o; }

struct MaterialOcclusionTextureInfo {
	std::size_t index;
	double strength;
	std::size_t texCoord;
	MaterialOcclusionTextureInfo(const Json::Value &o)
		: index(as<std::size_t>(o["index"]))
		, strength(1)
		, texCoord(0)
	{
		CHECK(o.isObject());
		if (o.isMember("strength")) {
			strength = as<double>(o["strength"]);
			CHECK(strength >= 0);
			CHECK(strength <= 1);
		}
		if (o.isMember("texCoord")) {
			texCoord = as<std::size_t>(o["texCoord"]);
		}
	}
};
template<> MaterialOcclusionTextureInfo as(const Json::Value &o) { return o; }

struct MaterialPbrMetallicRoughness {
	std::array<double, 4> baseColorFactor;
	std::optional<TextureInfo> baseColorTexture;
	double metallicFactor;
	std::optional<TextureInfo> metallicRoughnessTexture;
	double roughnessFactor;
	MaterialPbrMetallicRoughness(const Json::Value &o)
		: baseColorFactor{1, 1, 1, 1}
		, metallicFactor(1)
		, roughnessFactor(1)
	{
		CHECK(o.isObject());
		if (o.isMember("baseColorFactor")) {
			baseColorFactor = asArr<double, 4>(o["baseColorFactor"]);
			for (auto v: baseColorFactor) {
				CHECK(v >= 0);
				CHECK(v <= 1);
			}
		}
		if (o.isMember("baseColorTexture")) {
			baseColorTexture = as<TextureInfo>(o["baseColorTexture"]);
		}
		if (o.isMember("metallicFactor")) {
			metallicFactor = as<double>(o["metallicFactor"]);
			CHECK(metallicFactor >= 0);
			CHECK(metallicFactor <= 1);
		}
		if (o.isMember("metallicRoughnessTexture")) {
			metallicRoughnessTexture = as<TextureInfo>(o["metallicRoughnessTexture"]);
		}
		if (o.isMember("roughnessFactor")) {
			roughnessFactor = as<double>(o["roughnessFactor"]);
			CHECK(roughnessFactor >= 0);
			CHECK(roughnessFactor <= 1);
		}
	}
};
template<> MaterialPbrMetallicRoughness as(const Json::Value &o) { return o; }

struct Material {
	double alphaCutoff;
	enum class AlphaMode {
		BLEND,
		MASK,
		OPAQUE,
	};
	AlphaMode alphaMode;
	bool doubleSided;
	std::array<double, 3> emissiveFactor;
	std::optional<TextureInfo> emissiveTexture;
	std::optional<std::string> name;
	std::optional<MaterialNormalTextureInfo> normalTexture;
	std::optional<MaterialOcclusionTextureInfo> occlusionTexture;
	std::optional<MaterialPbrMetallicRoughness> pbrMetallicRoughness;
	Material(const Json::Value &o)
		: alphaCutoff(0.5)
		, alphaMode(AlphaMode::OPAQUE)
		, doubleSided(false)
		, emissiveFactor{0, 0, 0}
	{
		CHECK(o.isObject());
		if (o.isMember("alphaCutoff")) {
			alphaCutoff = as<double>(o["alphaCutoff"]);
			CHECK(alphaCutoff >= 0);
		}
		if (o.isMember("alphaMode")){
			static std::unordered_map<Json::String, AlphaMode> map = {
				{"BLEND", AlphaMode::BLEND},
				{"MASK", AlphaMode::MASK},
				{"OPAQUE", AlphaMode::OPAQUE},
			};
			auto v = o["alphaMode"]; CHECK(v.isString());
			alphaMode = map.at(v.asString());
		}
		if (o.isMember("doubleSided")) {
			doubleSided = as<bool>(o["doubleSided"]);
		}
		if (o.isMember("emissiveFactor")) {
			emissiveFactor = asArr<double, 3>(o["emissiveFactor"]);
			for (auto v: emissiveFactor) {
				CHECK(v >= 0);
				CHECK(v <= 1);
			}
		}
		if (o.isMember("emissiveTexture")) {
			emissiveTexture = as<TextureInfo>(o["emissiveTexture"]);
		}
		if (o.isMember("name")) {
			name = as<std::string>(o["name"]);
		}
		if (o.isMember("normalTexture")) {
			normalTexture = as<MaterialNormalTextureInfo>(o["normalTexture"]);
		}
		if (o.isMember("occlusionTexture")) {
			occlusionTexture = as<MaterialOcclusionTextureInfo>(o["occlusionTexture"]);
		}
		if (o.isMember("pbrMetallicRoughness")) {
			pbrMetallicRoughness = as<MaterialPbrMetallicRoughness>(o["pbrMetallicRoughness"]);
		}
	}
};
template<> Material as(const Json::Value &o) { return o; }

struct MeshPrimitive {
	static void enumeratedProps(const Json::Value &o, const std::string &name, std::optional<std::vector<std::size_t>> &attr) {
		for (std::size_t i = 0;; ++i) {
			const std::string s = name + "_" + std::to_string(i);
			if (!o.isMember(s)) break;
			if (i == 0) {
				attr = std::vector<std::size_t>();
			}
			attr->push_back(as<std::size_t>(o[s]));
		}
	}
	struct Attributes {
		std::optional<std::size_t> position, normal, tangent;
		std::optional<std::vector<std::size_t>> texcoord, color, joints, weights;
		Attributes(const Json::Value &o) {
			if (o.isMember("POSITION"))
				position = as<std::size_t>(o["POSITION"]);
			if (o.isMember("NORMAL"))
				normal = as<std::size_t>(o["NORMAL"]);
			if (o.isMember("TANGENT"))
				tangent = as<std::size_t>(o["TANGENT"]);
			enumeratedProps(o, "TEXCOORD", texcoord);
			enumeratedProps(o, "COLOR", color);
			enumeratedProps(o, "JOINTS", joints);
			enumeratedProps(o, "WEIGHTS", weights);
			CHECK(position.has_value()
					|| normal.has_value()
					|| tangent.has_value()
					|| texcoord.has_value()
					|| color.has_value()
					|| joints.has_value()
					|| weights.has_value());
		}
	};
	Attributes attributes;
	std::optional<std::size_t> indices;
	std::optional<std::size_t> material;
	enum class Mode {
		POINTS,
		LINES,
		LINE_LOOP,
		LINE_STRIP,
		TRIANGLES,
		TRIANGLE_STRIP,
		TRIANGLE_FAN,
	};
	Mode mode;
	struct MorphTargets {
		std::optional<std::size_t> position, normal, tangent;
		std::optional<std::vector<std::size_t>> texcoord, color;
		MorphTargets(const Json::Value &o) {
			if (o.isMember("POSITION"))
				position = as<std::size_t>(o["POSITION"]);
			if (o.isMember("NORMAL"))
				normal = as<std::size_t>(o["NORMAL"]);
			if (o.isMember("TANGENT"))
				tangent = as<std::size_t>(o["TANGENT"]);
			enumeratedProps(o, "TEXCOORD", texcoord);
			enumeratedProps(o, "COLOR", color);
			CHECK(position.has_value()
					|| normal.has_value()
					|| tangent.has_value()
					|| texcoord.has_value()
					|| color.has_value());
		}
	};
	std::optional<std::vector<MorphTargets>> targets;
	MeshPrimitive(const Json::Value &o)
		: attributes(Attributes(o["attributes"]))
		, mode(Mode::TRIANGLES)
	{
		CHECK(o.isObject());
		if (o.isMember("indices")) {
			indices = as<std::size_t>(o["indices"]);
		}
		if (o.isMember("material")) {
			material = as<std::size_t>(o["material"]);
		}
		if (o.isMember("mode")) {
			static std::unordered_map<Json::UInt64, Mode> map = {
				{0, Mode::POINTS},
				{1, Mode::LINES},
				{2, Mode::LINE_LOOP},
				{3, Mode::LINE_STRIP},
				{4, Mode::TRIANGLES},
				{5, Mode::TRIANGLE_STRIP},
				{6, Mode::TRIANGLE_FAN},
			};
			auto v = o["mode"]; CHECK(v.isUInt64());
			mode = map.at(v.asUInt64());
		}
		if (o.isMember("targets")) {
			targets = asVec<MorphTargets>(o["targets"]);
			CHECK(targets->size() >= 1);
		}
	}
};
template<> MeshPrimitive::MorphTargets as(const Json::Value &o) { return o; }
template<> MeshPrimitive as(const Json::Value &o) { return o; }

struct Mesh {
	std::optional<std::string> name;
	std::vector<MeshPrimitive> primitives;
	std::optional<std::vector<double>> weights;
	Mesh(const Json::Value &o)
		: primitives(asVec<MeshPrimitive>(o["primitives"]))
	{
		CHECK(o.isObject());
		if (o.isMember("name")) {
			name = as<std::string>(o["name"]);
		}
		CHECK(primitives.size() >= 1);
		if (o.isMember("weights")) {
			weights = asVec<double>(o["weights"]);
			CHECK(weights->size() >= 1);
		}
	}
};
template<> Mesh as(const Json::Value &o) { return o; }

struct Node {
	std::optional<std::size_t> camera;
	std::optional<std::vector<std::size_t>> children;
	typedef std::array<double, 16> Matrix;
	struct TRS {
		std::array<double, 3> translation;
		std::array<double, 4> rotation;
		std::array<double, 3> scale;
	};
	std::variant<Matrix, TRS> transform;
	std::optional<std::size_t> mesh;
	std::optional<std::string> name;
	std::optional<std::size_t> skin;
	std::optional<std::vector<double>> weights;
	Node(const Json::Value &o)
		: transform(Matrix {
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1
		})
	{
		CHECK(o.isObject());
		if (o.isMember("camera")) {
			camera = as<std::size_t>(o["camera"]);
		}
		if (o.isMember("children")) {
			children = asVec<std::size_t>(o["children"]);
			CHECK(children->size() >= 1);
			CHECK(std::unordered_set(children->begin(), children->end()).size() == children->size());
		}
		bool hasTRS = o.isMember("translation") || o.isMember("rotation") || o.isMember("scale");
		if (o.isMember("matrix")) {
			CHECK(!hasTRS);
			transform = asArr<double, 16>(o["matrix"]);
		} else if (hasTRS) {
			TRS trs;
			if (o.isMember("translation")) {
				trs.translation = asArr<double, 3>(o["translation"]);
			}
			if (o.isMember("rotation")) {
				trs.rotation = asArr<double, 4>(o["rotation"]);
				for (auto v: trs.rotation) {
					CHECK(v >= -1);
					CHECK(v <= 1);
				}
			}
			if (o.isMember("scale")) {
				trs.scale = asArr<double, 3>(o["scale"]);
			}
			transform = std::move(trs);
		}
		if (o.isMember("mesh")) {
			mesh = as<std::size_t>(o["mesh"]);
		}
		if (o.isMember("name")) {
			name = as<std::string>(o["name"]);
		}
		if (o.isMember("skin")) {
			skin = as<std::size_t>(o["skin"]);
		}
		if (o.isMember("weights")) {
			weights = asVec<double>(o["weights"]);
			CHECK(weights->size() >= 1);
		}
	}
};
template<> Node as(const Json::Value &o) { return o; }

struct Sampler {
	enum class MagFilter {
		NEAREST,
		LINEAR,
	};
	std::optional<MagFilter> magFilter;
	enum class MinFilter {
		NEAREST,
		LINEAR,
		NEAREST_MIPMAP_NEAREST,
		LINEAR_MIPMAP_NEAREST,
		NEAREST_MIPMAP_LINEAR,
		LINEAR_MIPMAP_LINEAR,
	};
	std::optional<MinFilter> minFilter;
	std::optional<std::string> name;
	enum class WrapS {
		REPEAT,
		CLAMP_TO_EDGE,
		MIRRORED_REPEAT,
	};
	WrapS wrapS;
	enum class WrapT {
		REPEAT,
		CLAMP_TO_EDGE,
		MIRRORED_REPEAT,
	};
	WrapT wrapT;
	Sampler(const Json::Value &o)
		: wrapS(WrapS::REPEAT)
		, wrapT(WrapT::REPEAT)
	{
		CHECK(o.isObject());
		if (o.isMember("magFilter")) {
			static std::unordered_map<Json::UInt64, MagFilter> map = {
				{9728, MagFilter::NEAREST},
				{9729, MagFilter::LINEAR},
			};
			auto v = o["magFilter"]; CHECK(v.isUInt64());
			magFilter = map.at(v.asUInt64());
		}
		if (o.isMember("minFilter")) {
			static std::unordered_map<Json::UInt64, MinFilter> map = {
				{9728, MinFilter::NEAREST},
				{9729, MinFilter::LINEAR},
				{9984, MinFilter::NEAREST_MIPMAP_NEAREST},
				{9985, MinFilter::LINEAR_MIPMAP_NEAREST},
				{9986, MinFilter::NEAREST_MIPMAP_LINEAR},
				{9987, MinFilter::LINEAR_MIPMAP_LINEAR},
			};
			auto v = o["minFilter"]; CHECK(v.isUInt64());
			minFilter = map.at(v.asUInt64());
		}
		if (o.isMember("name")) {
			name = as<std::string>(o["name"]);
		}
		if (o.isMember("wrapS")) {
			static std::unordered_map<Json::UInt64, WrapS> map = {
				{10497, WrapS::REPEAT},
				{33071, WrapS::CLAMP_TO_EDGE},
				{33648, WrapS::MIRRORED_REPEAT},
			};
			auto v = o["wrapS"]; CHECK(v.isUInt64());
			wrapS = map.at(v.asUInt64());
		}
		if (o.isMember("wrapT")) {
			static std::unordered_map<Json::UInt64, WrapT> map = {
				{10497, WrapT::REPEAT},
				{33071, WrapT::CLAMP_TO_EDGE},
				{33648, WrapT::MIRRORED_REPEAT},
			};
			auto v = o["wrapT"]; CHECK(v.isUInt64());
			wrapT = map.at(v.asUInt64());
		}
	}
};
template<> Sampler as(const Json::Value &o) { return o; }

struct Scene {
	std::optional<std::string> name;
	std::optional<std::vector<std::size_t>> nodes;
	Scene(const Json::Value &o)
	{
		CHECK(o.isObject());
		if (o.isMember("name")) {
			name = as<std::string>(o["name"]);
		}
		if (o.isMember("nodes")) {
			nodes = asVec<std::size_t>(o["nodes"]);
			CHECK(nodes->size() >= 1);
			CHECK(std::unordered_set(nodes->begin(), nodes->end()).size() == nodes->size());
		}
	}
};
template<> Scene as(const Json::Value &o) { return o; }

struct Skin {
	std::optional<std::size_t> inverseBindMatrices;
	std::vector<std::size_t> joints;
	std::optional<std::string> name;
	std::optional<std::size_t> skeleton;
	Skin(const Json::Value &o)
		: joints(asVec<std::size_t>(o["joints"]))
	{
		CHECK(o.isObject());
		if (o.isMember("inverseBindMatrices")) {
			inverseBindMatrices = as<std::size_t>(o["inverseBindMatrices"]);
		}
		CHECK(joints.size() >= 1);
		CHECK(std::unordered_set(joints.begin(), joints.end()).size() == joints.size());
		if (o.isMember("name")) {
			name = as<std::string>(o["name"]);
		}
		if (o.isMember("skeleton")) {
			skeleton = as<std::size_t>(o["skeleton"]);
		}
	}
};
template<> Skin as(const Json::Value &o) { return o; }

struct Texture {
	std::optional<std::string> name;
	std::optional<std::size_t> sampler;
	std::optional<std::size_t> source;
	Texture(const Json::Value &o)
	{
		CHECK(o.isObject());
		if (o.isMember("name")) {
			name = as<std::string>(o["name"]);
		}
		if (o.isMember("sampler")) {
			sampler = as<std::size_t>(o["sampler"]);
		}
		if (o.isMember("source")) {
			source = as<std::size_t>(o["source"]);
		}
	}
};
template<> Texture as(const Json::Value &o) { return o; }

struct GlTF {
	std::optional<std::vector<Accessor>> accessors;
	std::optional<std::vector<Animation>> animations;
	Asset asset;
	std::optional<std::vector<BufferView>> bufferViews;
	std::optional<std::vector<Buffer>> buffers;
	std::optional<std::vector<Camera>> cameras;
	std::optional<std::vector<std::string>> extensionsRequired;
	std::optional<std::vector<std::string>> extensionsUsed;
	std::optional<std::vector<Image>> images;
	std::optional<std::vector<Material>> materials;
	std::optional<std::vector<Mesh>> meshes;
	std::optional<std::vector<Node>> nodes;
	std::optional<std::vector<Sampler>> samplers;
	std::optional<std::size_t> scene;
	std::optional<std::vector<Scene>> scenes;
	std::optional<std::vector<Skin>> skins;
	std::optional<std::vector<Texture>> textures;
	static std::vector<unsigned char> uriError(const std::string &uri) {
		// only base64 data URI support by default
		throw std::runtime_error("unsupported URI: " + uri);
	}
	GlTF(const Json::Value &o,
			const std::function<std::vector<unsigned char>(const std::string &uri)> &resolveURI = uriError)
		: asset(as<Asset>(o["asset"]))
	{
		CHECK(o.isObject());
		if (o.isMember("accessors")) {
			accessors = asVec<Accessor>(o["accessors"]);
			CHECK(accessors->size() >= 1);
		}
		if (o.isMember("animations")) {
			animations = asVec<Animation>(o["animations"]);
			CHECK(animations->size() >= 1);
		}
		if (o.isMember("bufferViews")) {
			bufferViews = asVec<BufferView>(o["bufferViews"]);
			CHECK(bufferViews->size() >= 1);
		}
		if (o.isMember("buffers")) {
			auto b = o["buffers"];
			CHECK(b.isArray());
			std::vector<Buffer> bufs;
			bufs.reserve(b.size());
			for (Json::ArrayIndex i = 0; i < b.size(); ++i) {
				bufs.push_back(Buffer(b[i], resolveURI));
			}
			CHECK(bufs.size() >= 1);
			buffers = std::move(bufs);
		}
		if (o.isMember("cameras")) {
			cameras = asVec<Camera>(o["cameras"]);
			CHECK(cameras->size() >= 1);
		}
		if (o.isMember("extensionsRequired")) {
			extensionsRequired = asVec<std::string>(o["extensionsRequired"]);
			CHECK(extensionsRequired->size() >= 1);
			CHECK(std::unordered_set(extensionsRequired->begin(), extensionsRequired->end()).size() == extensionsRequired->size());
		}
		if (o.isMember("extensionsUsed")) {
			extensionsUsed = asVec<std::string>(o["extensionsUsed"]);
			CHECK(extensionsUsed->size() >= 1);
			CHECK(std::unordered_set(extensionsUsed->begin(), extensionsUsed->end()).size() == extensionsUsed->size());
		}
		if (o.isMember("images")) {
			images = asVec<Image>(o["images"]);
			CHECK(images->size() >= 1);
		}
		if (o.isMember("materials")) {
			materials = asVec<Material>(o["materials"]);
			CHECK(materials->size() >= 1);
		}
		if (o.isMember("meshes")) {
			meshes = asVec<Mesh>(o["meshes"]);
			CHECK(meshes->size() >= 1);
		}
		if (o.isMember("nodes")) {
			nodes = asVec<Node>(o["nodes"]);
			CHECK(nodes->size() >= 1);
			// Nodes must be a forest
			std::vector<bool> visited(nodes->size());
			std::vector<bool> roots(nodes->size());
			for (std::size_t i = 0; i < nodes->size(); ++i) {
				if (visited[i]) continue;
				std::stack<std::size_t, std::vector<std::size_t>> toVisit;
				toVisit.push(i);
				roots.at(i) = true;
				do {
					std::size_t j = toVisit.top();
					toVisit.pop();
					auto children = nodes->at(j).children;
					if (!children.has_value()) continue;
					for (auto child : children.value()) {
						toVisit.push(child);
						if (roots.at(child)) {
							roots.at(child) = false;
						} else {
							CHECK(!visited[child]);
						}
						visited[child] = true;
					}
				} while (!toVisit.empty());
			}
		}
		if (o.isMember("samplers")) {
			samplers = asVec<Sampler>(o["samplers"]);
			CHECK(samplers->size() >= 1);
		}
		if (o.isMember("scene")) {
			scene = as<std::size_t>(o["scene"]);
		}
		if (o.isMember("scenes")) {
			scenes = asVec<Scene>(o["scenes"]);
			CHECK(scenes->size() >= 1);
		}
		if (o.isMember("skins")) {
			skins = asVec<Skin>(o["skins"]);
			CHECK(skins->size() >= 1);
		}
		if (o.isMember("textures")) {
			textures = asVec<Texture>(o["textures"]);
			CHECK(textures->size() >= 1);
		}

		// Validation

		checkForall(bufferViews, [=](const BufferView &view) {
			CHECK(buffers.has_value());
			const Buffer &buf = buffers->at(view.buffer);
			// Be careful because of possible integer overflows.
			CHECK(view.byteOffset < buf.byteLength);
			CHECK(view.byteLength < buf.byteLength);
			CHECK(view.byteOffset <= buf.byteLength - view.byteLength);
		});

		checkForall(accessors, [=](const Accessor &accessor) {
			checkIndex(bufferViews, accessor.bufferView);
			if (accessor.bufferView.has_value()) {
				const BufferView &view = bufferViews->at(accessor.bufferView.value());
				if (view.byteStride.has_value())
					CHECK(view.byteStride.value() % accessor.componentSize(accessor.componentType) == 0);
				CHECK(accessor.byteOffset < view.byteLength);
				// Use division to avoid overflows.
				CHECK(accessor.count <= view.byteLength / accessor.elementSize());
			}
			if (accessor.sparse.has_value()) {
				// Again be careful because of possible integer overflows.
				{
					const auto &indices = accessor.sparse->indices;
					checkIndex(bufferViews, indices.bufferView);
					const BufferView &view = bufferViews->at(indices.bufferView);
					CHECK(indices.byteOffset < view.byteLength);
					CHECK(accessor.sparse->count <= (view.byteLength - indices.byteOffset) / indices.elementSize());
				}
				{
					const auto &values = accessor.sparse->values;
					checkIndex(bufferViews, values.bufferView);
					const BufferView &view = bufferViews->at(values.bufferView);
					CHECK(values.byteOffset < view.byteLength);
					CHECK(accessor.sparse->count <= (view.byteLength - values.byteOffset) / accessor.elementSize());
				}
			}
		});

		checkForall(images, [=](const Image &image) {
			checkIndex(bufferViews, image.bufferView);
		});

		checkForall(meshes, [=](const Mesh &mesh) {
			for (const auto &primitive : mesh.primitives) {
				checkIndex(accessors, primitive.indices);
				checkIndex(materials, primitive.material);
				checkIndex(accessors, primitive.attributes.normal);
				checkIndex(accessors, primitive.attributes.position);
				checkIndex(accessors, primitive.attributes.tangent);
				checkForall(primitive.attributes.texcoord, [=](const std::size_t &i) {
					checkIndex(accessors, i);
				});
				checkForall(primitive.attributes.color, [=](const std::size_t &i) {
					checkIndex(accessors, i);
				});
				checkForall(primitive.attributes.joints, [=](const std::size_t &i) {
					checkIndex(accessors, i);
				});
				checkForall(primitive.attributes.weights, [=](const std::size_t &i) {
					checkIndex(accessors, i);
				});
				if (primitive.material.has_value()) {
					const Material &material = materials->at(primitive.material.value());
					if (material.emissiveTexture.has_value()) {
						CHECK(primitive.attributes.texcoord.has_value());
						CHECK(material.emissiveTexture->texCoord < primitive.attributes.texcoord->size());
					}
					if (material.normalTexture.has_value()) {
						CHECK(primitive.attributes.texcoord.has_value());
						CHECK(material.normalTexture->texCoord < primitive.attributes.texcoord->size());
					}
					if (material.occlusionTexture.has_value()) {
						CHECK(primitive.attributes.texcoord.has_value());
						CHECK(material.occlusionTexture->texCoord < primitive.attributes.texcoord->size());
					}
				}
				checkForall(primitive.targets, [=](const MeshPrimitive::MorphTargets &target) {
					checkIndex(accessors, target.normal);
					checkIndex(accessors, target.position);
					checkIndex(accessors, target.tangent);
					checkForall(target.texcoord, [=](const std::size_t &i) {
						checkIndex(accessors, i);
					});
					checkForall(target.color, [=](const std::size_t &i) {
						checkIndex(accessors, i);
					});
				});
			}
		});

		checkForall(nodes, [=](const Node &node) {
			checkIndex(cameras, node.camera);
			checkIndex(meshes, node.mesh);
			checkIndex(skins, node.skin);
		});

		checkForall(scenes, [=](const Scene &scene) {
			checkForall(scene.nodes, [=](const size_t &i) {
				checkIndex(nodes, i);
			});
		});

		checkForall(skins, [=](const Skin &skin) {
			checkIndex(accessors, skin.inverseBindMatrices);
			for (const std::size_t &i : skin.joints)
				checkIndex(nodes, i);
			checkIndex(nodes, skin.skeleton);
		});

		checkForall(textures, [=](const Texture &texture) {
			checkIndex(samplers, texture.sampler);
			checkIndex(images, texture.source);
		});

		checkForall(animations, [=](const Animation &animation) {
			for (const auto &sampler : animation.samplers) {
				checkIndex(accessors, sampler.input);
				const auto &accessor = accessors->at(sampler.input);
				CHECK(accessor.type == Accessor::Type::SCALAR);
				CHECK(accessor.componentType == Accessor::ComponentType::FLOAT);
				checkIndex(accessors, sampler.output);
			}
			for (const auto &channel : animation.channels) {
				checkIndex(nodes, channel.target.node);
				checkIndex(animation.samplers, channel.sampler);
			}
		});

		checkIndex(scenes, scene);
	}
};

}
#pragma once

#include <vector>
#include <unordered_map>
#include "Common/Vulkan/VulkanContext.h"
//#include "Common/Log.h"

// VulkanMemory
//
// Vulkan memory management utils.

// VulkanPushBuffer
// Simple incrementing allocator.
// Use these to push vertex, index and uniform data. Generally you'll have two of these
// and alternate on each frame. Make sure not to reset until the fence from the last time you used it
// has completed.
//
// TODO: Make it possible to suballocate pushbuffers from a large DeviceMemory block.
class VulkanPushBuffer {
	struct BufInfo {
		size_t size;
		VkBuffer buffer;
		VmaAllocation allocation;
	};

public:
	VulkanPushBuffer(VulkanContext *vulkan, VkBufferUsageFlags usage)
		: vulkan_(vulkan), usage_(usage), buf_(0), offset_(0), writePtr_(nullptr) {
		if (usage_ == VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) {
			magnify_ = 6;
		}
		else {
			magnify_ = 4;
		}
	}

	~VulkanPushBuffer() {
		assert(buffers_.empty());
	}

	void Destroy() {
		if (writePtr_) {
			Unmap();
		}
		for (BufInfo &info : buffers_) {
			vulkan_->FreeBuffer(&info.buffer, &info.allocation);
		}
		buffers_.clear();
	}

	// Needs context in case of defragment.
	void Begin() {
		Destroy();
		buf_ = 0;
		offset_ = 0;
		// Note: we must defrag because some buffers may be smaller than size_.
		//Defragment();
	}

	void BeginNoReset() {
		
	}

	void End() {
		if (writePtr_) {
			Unmap();
		}
	}

	void Map() {
		//_dbg_assert_(G3D, !writePtr_);
		VkResult res = vulkan_->MapMemory(buffers_[buf_].allocation, (void **)(&writePtr_));
		//_dbg_assert_(G3D, writePtr_);
		assert(VK_SUCCESS == res);
	}

	void Unmap() {
		//_dbg_assert_(G3D, writePtr_ != 0);
		vulkan_->UnmapMemory(buffers_[buf_].allocation);
		writePtr_ = nullptr;
	}

	// When using the returned memory, make sure to bind the returned vkbuf.
	// This will later allow for handling overflow correctly.
	size_t Allocate(size_t size, VkBuffer *vkbuf) {
		size_t out = offset_;
		offset_ += (size + 3) & ~3;  // Round up to 4 bytes.
		if (buffers_.empty() || offset_ >= buffers_[buf_].size) {
			*vkbuf = AddBuffer(size << magnify_);
			out = offset_;
			offset_ += (size + 3) & ~3;  // Round up to 4 bytes.
		}
		else {
			*vkbuf = buffers_[buf_].buffer;
		}
		return out;
	}

	// Returns the offset that should be used when binding this buffer to get this data.
	size_t Push(const void *data, size_t size, VkBuffer *vkbuf) {
		size_t off = Allocate(size, vkbuf);
		memcpy(writePtr_ + off, data, size);
		return off;
	}

	uint32_t PushAligned(const void *data, size_t size, int align, VkBuffer *vkbuf) {
		offset_ = (offset_ + align - 1) & ~(align - 1);
		size_t off = Allocate(size, vkbuf);
		memcpy(writePtr_ + off, data, size);
		return (uint32_t)off;
	}


	// "Zero-copy" variant - you can write the data directly as you compute it.
	// Recommended.
	void *Push(size_t size, uint32_t *bindOffset, VkBuffer *vkbuf) {
		size_t off = Allocate(size, vkbuf);
		*bindOffset = (uint32_t)off;
		return writePtr_ + off;
	}

	void *PushAligned(size_t size, uint32_t *bindOffset, VkBuffer *vkbuf, int align) {
		offset_ = (offset_ + align - 1) & ~(align - 1);
		size_t off = Allocate(size, vkbuf);
		*bindOffset = (uint32_t)off;
		return writePtr_ + off;
	}

	size_t GetTotalSize() const {
		size_t sum = 0;
		for (auto buf : buffers_) {
			sum += buf.size;
		}
		sum += offset_;
		return sum;
	}

private:
	VkBuffer AddBuffer(size_t size) {
		VkBufferCreateInfo b{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		b.size = size;
		b.flags = 0;
		b.usage = usage_;
		b.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		b.queueFamilyIndexCount = 0;
		b.pQueueFamilyIndices = nullptr;

		BufInfo info;
		VkResult res = vulkan_->AllocPushBuffer(b, &info.buffer, &info.allocation);
		if (VK_SUCCESS != res) {
			//_assert_msg_(G3D, false, "vkCreateBuffer failed! result=%d", (int)res);
			return nullptr;
		}

		if (writePtr_ != nullptr)
		{
			Unmap();
		}

		offset_ = 0;
		info.size = size;
		buf_ = buffers_.size();
		buffers_.push_back(info);

		Map();

		return info.buffer;
	}

	VulkanContext *vulkan_;
	VkBufferUsageFlags usage_;
	std::vector<BufInfo> buffers_;
	size_t buf_;
	size_t offset_;
	uint8_t *writePtr_;
	size_t magnify_;
};

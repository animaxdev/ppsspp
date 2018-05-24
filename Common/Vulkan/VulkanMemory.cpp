// Copyright (c) 2016- PPSSPP Project.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2.0 or later versions.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License 2.0 for more details.

// A copy of the GPL 2.0 should have been included with the program.
// If not, see http://www.gnu.org/licenses/

// Official git repository and contact information can be found at
// https://github.com/hrydgard/ppsspp and http://www.ppsspp.org/.

// Additionally, Common/Vulkan/* , including this file, are also licensed
// under the public domain.

#include "Common/Log.h"
#include "Common/Vulkan/VulkanMemory.h"
#include "base/timeutil.h"
#include "math/math_util.h"

VulkanPushBuffer::VulkanPushBuffer(VulkanContext *vulkan, VkBufferUsageFlags usage, size_t size)
	: vulkan_(vulkan), usage_(usage), buf_(0), offset_(0), size_(size), writePtr_(nullptr) {
	bool res = AddBuffer();
	assert(res);
}

VulkanPushBuffer::~VulkanPushBuffer() {
	assert(buffers_.empty());
}

bool VulkanPushBuffer::AddBuffer() {
	BufInfo info;

	VkBufferCreateInfo b{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	b.size = size_;
	b.flags = 0;
	b.usage = usage_;
	b.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	b.queueFamilyIndexCount = 0;
	b.pQueueFamilyIndices = nullptr;


	VkResult res = vulkan_->AllocBuffer(b, &info.buffer, &info.allocation);
	if (VK_SUCCESS != res) {
		_assert_msg_(G3D, false, "vkCreateBuffer failed! result=%d", (int)res);
		return false;
	}

	buffers_.push_back(info);
	buf_ = buffers_.size() - 1;
	return true;
}

void VulkanPushBuffer::Destroy() {
	for (BufInfo &info : buffers_) {
		vulkan_->FreeBuffer(&info.buffer, &info.allocation);
	}
	buffers_.clear();
}

void VulkanPushBuffer::NextBuffer(size_t minSize) {
	// First, unmap the current memory.
	Unmap();

	buf_++;
	if (buf_ >= buffers_.size() || minSize > size_) {
		// Before creating the buffer, adjust to the new size_ if necessary.
		while (size_ < minSize) {
			size_ <<= 1;
		}

		bool res = AddBuffer();
		assert(res);
		if (!res) {
			// Let's try not to crash at least?
			buf_ = 0;
		}
	}

	// Now, move to the next buffer and map it.
	offset_ = 0;
	Map();
}

size_t VulkanPushBuffer::GetTotalSize() const {
	size_t sum = 0;
	if (buffers_.size() > 1)
		sum += size_ * (buffers_.size() - 1);
	sum += offset_;
	return sum;
}

void VulkanPushBuffer::Map() {
	_dbg_assert_(G3D, !writePtr_);
	VkResult res = vulkan_->MapMemory(buffers_[buf_].allocation, (void **)(&writePtr_));
	_dbg_assert_(G3D, writePtr_);
	assert(VK_SUCCESS == res);
}

void VulkanPushBuffer::Unmap() {
	_dbg_assert_(G3D, writePtr_ != 0);
	vulkan_->UnmapMemory(buffers_[buf_].allocation);
	writePtr_ = nullptr;
}


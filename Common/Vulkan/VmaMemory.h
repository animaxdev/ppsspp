#pragma once

#ifndef __VMA_MEMORY_H__
#define __VMA_MEMORY_H__

#include "ext/vulkan/vulkan.h"
#include "Common/Vulkan/vk_mem_alloc.h"

class VmaMemoryAllocator
{
public:
	using mem_handle_t = void *;

	VmaMemoryAllocator(VkDevice dev, VkPhysicalDevice pdev);
	~VmaMemoryAllocator();

	void Destroy();

	mem_handle_t  Alloc(size_t block_sz, uint32_t memory_type_index);

	void Free(mem_handle_t mem_handle);

	void* Map(mem_handle_t mem_handle, size_t offset, size_t size);

	void Unmap(mem_handle_t mem_handle);

	VkDeviceMemory GetDeviceMemory(mem_handle_t mem_handle);

	size_t GetOffset(mem_handle_t mem_handle);

private:
	VmaAllocator m_allocator;
};




#endif // __VMA_MEMORY_H__
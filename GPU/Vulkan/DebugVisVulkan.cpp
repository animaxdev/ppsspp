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

#include "gfx_es2/draw_buffer.h"
#include "thin3d/thin3d.h"
#include "ui/ui_context.h"
#include "ui/view.h"

#include "DebugVisVulkan.h"
#include "Common/Vulkan/VulkanMemory.h"
#include "Common/Vulkan/VulkanImage.h"
#include "GPU/Vulkan/GPU_Vulkan.h"
#include "GPU/Vulkan/VulkanUtil.h"

void DrawAllocatorVis(UIContext *ui, GPUInterface *gpu) {
	if (!gpu) {
		return;
	}
}

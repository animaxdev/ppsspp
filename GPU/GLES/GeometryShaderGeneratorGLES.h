#pragma once

#include <stdint.h>
#include "Common/CommonTypes.h"

struct VShaderID;

void GenerateGeometryShader(const VShaderID &id, char *buffer, uint32_t *attrMask, uint64_t *uniformMask);

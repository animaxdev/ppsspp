#pragma once

class UIContext;

#ifdef USE_PROFILER

void DrawProfile(UIContext &ui);

#endif

#ifdef VKSTEP_PROFILER

void DrawVKStepProfile(UIContext &ui);

#endif
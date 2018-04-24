#pragma once

#include <cstdint>

// #define USE_PROFILER

#ifdef USE_PROFILER

class DrawBuffer;

void internal_profiler_init();
void internal_profiler_end_frame();

int internal_profiler_enter(const char *category_name, int *thread_id);  // Returns the category number.
void internal_profiler_leave(int thread_id, int category);

const char *Profiler_GetCategoryName(int i);
int Profiler_GetNumCategories();
int Profiler_GetHistoryLength();
int Profiler_GetNumThreads();
void Profiler_GetSlowestThreads(int *data, int count);
void Profiler_GetSlowestHistory(int category, int *slowestThreads, float *data, int count);
void Profiler_GetHistory(int category, int thread, float *data, int count);

class ProfileThis {
public:
	ProfileThis(const char *category) {
		cat_ = internal_profiler_enter(category, &thread_);
	}
	~ProfileThis() {
		internal_profiler_leave(thread_, cat_);
	}
private:
	int cat_;
	int thread_;
};

#define PROFILE_INIT() internal_profiler_init()
#define PROFILE_THIS_SCOPE(cat) ProfileThis _profile_scoped(cat)
#define PROFILE_END_FRAME() internal_profiler_end_frame()

#else

#define PROFILE_INIT()
#define PROFILE_THIS_SCOPE(cat)
#define PROFILE_END_FRAME()

#endif


// #define VKSTEP_PROFILER

#ifdef VKSTEP_PROFILER

#include "ext/native/thin3d/VulkanQueueRunner.h"

int VKStepProfiler_GetNumQueue();
int VKStepProfiler_GetNumSteps(int i);
void VKStepProfiler_GetStep(int queue_index, int step_index, int &type, const char * &name, double &elapsed);
void VKStepProfiler_RemoveQueue(int count);

class VKQueueProfiler {
public:
	VKQueueProfiler(const std::vector<VKRStep *> &steps);
	~VKQueueProfiler();
};


class VKStepProfiler {
public:
	VKStepProfiler(const VKRStep &step);
	~VKStepProfiler();
};

#define PROFILE_THIS_QUEUE(steps) VKQueueProfiler _profile_this_queue(steps)
#define PROFILE_THIS_STEP(step) VKStepProfiler _profile_this_step(step)

#else

#define PROFILE_THIS_QUEUE(steps)
#define PROFILE_THIS_STEP(step)

#endif

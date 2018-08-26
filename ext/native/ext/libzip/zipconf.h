#ifndef _HAD_ZIPCONF_H
#define _HAD_ZIPCONF_H

/*
   zipconf.h -- platform specific include file

   This file was generated automatically by CMake
   based on ../cmake-zipconf.h.in.
 */

#define LIBZIP_VERSION "1.5.1"
#define LIBZIP_VERSION_MAJOR 1
#define LIBZIP_VERSION_MINOR 4
#define LIBZIP_VERSION_MICRO 0

#define ZIP_STATIC
#define HAVE_CONFIG_H

#define HAVE_INTTYPES_H_LIBZIP
#define HAVE_STDINT_H_LIBZIP

#ifndef _WIN32
#define HAVE_SYS_TYPES_H_LIBZIP
#endif

#define HAVE_INT8_T_LIBZIP
#define HAVE_UINT8_T_LIBZIP

#define HAVE_INT16_T_LIBZIP
#define HAVE_UINT16_T_LIBZIP

#define HAVE_INT32_T_LIBZIP
#define HAVE_UINT32_T_LIBZIP

#define HAVE_INT64_T_LIBZIP
#define HAVE_UINT64_T_LIBZIP
#define HAVE_SSIZE_T_LIBZIP

#if defined(HAVE_STDINT_H_LIBZIP)
#include <stdint.h>
#elif defined(HAVE_INTTYPES_H_LIBZIP)
#include <inttypes.h>
#elif defined(HAVE_SYS_TYPES_H_LIBZIP)
#include <sys/types.h>
#endif

#if defined(HAVE_INT8_T_LIBZIP)
typedef int8_t zip_int8_t;
#elif defined(HAVE___INT8_LIBZIP)
typedef __int8 zip_int8_t;
#else
typedef signed char zip_int8_t;
#endif

#if defined(HAVE_UINT8_T_LIBZIP)
typedef uint8_t zip_uint8_t;
#else
typedef unsigned char zip_uint8_t;
#endif

#if defined(HAVE_INT16_T_LIBZIP)
typedef int16_t zip_int16_t;
#else
typedef signed short zip_int16_t;
#endif

#if defined(HAVE_UINT16_T_LIBZIP)
typedef uint16_t zip_uint16_t;
#else
typedef unsigned short zip_uint16_t;
#endif

#if defined(HAVE_INT32_T_LIBZIP)
typedef int32_t zip_int32_t;
#else
typedef signed long zip_int32_t;
#endif

#if defined(HAVE_UINT32_T_LIBZIP)
typedef uint32_t zip_uint32_t;
#else
typedef unsigned long zip_uint32_t;
#endif

#if defined(HAVE_INT64_T_LIBZIP)
typedef int64_t zip_int64_t;
#else
typedef signed long long zip_int64_t;
#endif

#if defined(HAVE_UINT64_T_LIBZIP)
typedef uint64_t zip_uint64_t;
#else
typedef unsigned long long zip_uint64_t;
#endif

#define ZIP_INT8_MIN	 (-ZIP_INT8_MAX-1)
#define ZIP_INT8_MAX	 0x7f
#define ZIP_UINT8_MAX	 0xff

#define ZIP_INT16_MIN	 (-ZIP_INT16_MAX-1)
#define ZIP_INT16_MAX	 0x7fff
#define ZIP_UINT16_MAX	 0xffff

#define ZIP_INT32_MIN	 (-ZIP_INT32_MAX-1L)
#define ZIP_INT32_MAX	 0x7fffffffL
#define ZIP_UINT32_MAX	 0xffffffffLU

#define ZIP_INT64_MIN	 (-ZIP_INT64_MAX-1LL)
#define ZIP_INT64_MAX	 0x7fffffffffffffffLL
#define ZIP_UINT64_MAX	 0xffffffffffffffffULL

#endif /* zipconf.h */

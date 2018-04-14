#ifndef HAD_CONFIG_H
#define HAD_CONFIG_H
#ifndef _HAD_ZIPCONF_H
#include "zipconf.h"
#endif

/* BEGIN DEFINES */
#define HAVE___PROGNAME
#define HAVE__CHMOD
#define HAVE__CLOSE
#define HAVE__DUP
#define HAVE__FDOPEN
#define HAVE__FILENO
#define HAVE__OPEN
#define HAVE__SETMODE

#define HAVE__STRDUP
#define HAVE__STRICMP
#define HAVE__STRTOI64
#define HAVE__STRTOUI64
#define HAVE__UMASK
#define HAVE__UNLINK
//#define HAVE_CLONEFILE
#define HAVE_FILENO
#define HAVE_FSEEKO
#define HAVE_FTELLO
#define HAVE_GETPROGNAME
//#define HAVE_LIBBZ2
#define HAVE_OPEN

#define HAVE_SETMODE
#define HAVE_SNPRINTF
#define HAVE_SSIZE_T_LIBZIP

#define HAVE_STRDUP
#define HAVE_STRICMP
#define HAVE_STRTOLL
#define HAVE_STRTOULL

#define HAVE_STDBOOL_H



#ifdef _WIN32
#define HAVE_MOVEFILEEXA
#endif

#ifndef _WIN32
#define HAVE_STRINGS_H
#define HAVE_STRCASECMP
#define HAVE_MKSTEMP
#define HAVE_STRUCT_TM_TM_ZONE
#define HAVE_UNISTD_H
#define HAVE__SNPRINTF
#endif

#define __INT8_LIBZIP ${__INT8_LIBZIP}
#define INT8_T_LIBZIP ${INT8_T_LIBZIP}
#define UINT8_T_LIBZIP ${UINT8_T_LIBZIP}
#define __INT16_LIBZIP ${__INT16_LIBZIP}
#define INT16_T_LIBZIP ${INT16_T_LIBZIP}
#define UINT16_T_LIBZIP ${UINT16_T_LIBZIP}
#define __INT32_LIBZIP ${__INT32_LIBZIP}
#define INT32_T_LIBZIP ${INT32_T_LIBZIP}
#define UINT32_T_LIBZIP ${UINT32_T_LIBZIP}
#define __INT64_LIBZIP ${__INT64_LIBZIP}
#define INT64_T_LIBZIP ${INT64_T_LIBZIP}
#define UINT64_T_LIBZIP ${UINT64_T_LIBZIP}
#define SIZEOF_OFF_T 8
#define SIZE_T_LIBZIP ${SIZE_T_LIBZIP}
#define SSIZE_T_LIBZIP ${SSIZE_T_LIBZIP}

#define HAVE_DIRENT_H
#define HAVE_FTS_H
#define HAVE_NDIR_H
#define HAVE_SYS_DIR_H
#define HAVE_SYS_NDIR_H
#define WORDS_BIGENDIAN
#define HAVE_SHARED

/* END DEFINES */
#define PACKAGE "libzip"
#define VERSION "1.4.0"

#ifndef HAVE_SSIZE_T_LIBZIP
#  if SIZE_T_LIBZIP == INT_LIBZIP
typedef int ssize_t;
#  elif SIZE_T_LIBZIP == LONG_LIBZIP
typedef long ssize_t;
#  elif SIZE_T_LIBZIP == LONG_LONG_LIBZIP
typedef long long ssize_t;
#  else
#error no suitable type for ssize_t found
#  endif
#endif

#endif /* HAD_CONFIG_H */

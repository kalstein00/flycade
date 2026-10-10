/* External POSIX boundary fault injection; never linked into the application. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int target(int fd) {
    char link[64], path[4096];
    snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
    ssize_t n = readlink(link, path, sizeof(path)-1);
    if (n < 0) return 0;
    path[n] = 0;
    const char *root = getenv("FLYCADE_FAULT_RUN");
    return root && !strncmp(path, root, strlen(root)) && strstr(path, "/checkpoints/") && strstr(path, ".partial");
}
ssize_t write(int fd, const void *buffer, size_t count) {
    ssize_t (*real_write)(int,const void*,size_t) = dlsym(RTLD_NEXT, "write");
    const char *mode = getenv("FLYCADE_FAULT_MODE");
    if (mode && !strcmp(mode,"enospc") && target(fd)) { errno = ENOSPC; return -1; }
    return real_write(fd,buffer,count);
}
int fsync(int fd) {
    int (*real_fsync)(int) = dlsym(RTLD_NEXT, "fsync");
    const char *mode = getenv("FLYCADE_FAULT_MODE");
    if (mode && !strcmp(mode,"fsync") && target(fd)) { errno = EIO; return -1; }
    return real_fsync(fd);
}
int rename(const char *old, const char *next) {
    int (*real_rename)(const char*,const char*) = dlsym(RTLD_NEXT, "rename");
    const char *mode = getenv("FLYCADE_FAULT_MODE"), *root = getenv("FLYCADE_FAULT_RUN");
    if (mode && root && !strcmp(mode,"publish") && !strncmp(next,root,strlen(root)) && strstr(next,"/latest.json")) { errno = EIO; return -1; }
    return real_rename(old,next);
}

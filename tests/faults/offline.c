/* Test-only network boundary: allow loopback, reject every external connection. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>
int connect(int fd, const struct sockaddr *address, socklen_t length) {
    int (*real_connect)(int,const struct sockaddr*,socklen_t)=dlsym(RTLD_NEXT,"connect");
    if (address && address->sa_family==AF_INET) {
        const struct sockaddr_in *v4=(const struct sockaddr_in*)address;
        if ((ntohl(v4->sin_addr.s_addr)>>24)!=127) {errno=ENETUNREACH;return -1;}
    }
    if (address && address->sa_family==AF_INET6) {
        const struct sockaddr_in6 *v6=(const struct sockaddr_in6*)address;
        if (!IN6_IS_ADDR_LOOPBACK(&v6->sin6_addr)) {errno=ENETUNREACH;return -1;}
    }
    return real_connect(fd,address,length);
}

/* Force bind(port=0) into a firewall-permitted TCP port range.
 *
 * Why this exists: the compute-node firewall only admits a narrow port window,
 * but NCCL's bootstrap sockets ask the kernel to pick a port (bind with port 0)
 * and the kernel draws from ip_local_port_range, which is mostly outside that
 * window. NCCL has no environment variable to constrain those ports, and
 * changing ip_local_port_range needs root. Interposing on bind() is the only
 * user-space lever.
 *
 * Build and use:
 *   gcc -O2 -fPIC -shared -o .tmp/port_shim.so scripts/port_shim.c -ldl
 *   PORT_SHIM_RANGE=61001-63000 LD_PRELOAD=$PWD/.tmp/port_shim.so <cmd>
 *   PORT_SHIM_VERBOSE=1 also logs every rewritten bind to stderr.
 *
 * Only sockets that would otherwise get an arbitrary port are touched: TCP,
 * port 0, non-loopback. Everything else, including an explicit MASTER_PORT, is
 * passed straight through. If no port in the range can be claimed, the original
 * bind is performed so behaviour degrades to the unshimmed default rather than
 * failing in a novel way.
 */

#define _GNU_SOURCE

#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

typedef int (*bind_fn)(int, const struct sockaddr *, socklen_t);

static bind_fn real_bind;
static unsigned lo_port, hi_port;
static int enabled;
static int verbose;

/* bind() can be reached from another library's constructor, i.e. before ours has
 * run, so the pointer is resolved on demand rather than trusted to be ready. */
static bind_fn next_bind(void)
{
    if (!real_bind)
        real_bind = (bind_fn)dlsym(RTLD_NEXT, "bind");
    return real_bind;
}

__attribute__((constructor)) static void shim_init(void)
{
    next_bind();
    verbose = getenv("PORT_SHIM_VERBOSE") != NULL;

    const char *range = getenv("PORT_SHIM_RANGE");
    unsigned lo, hi;
    if (real_bind && range && sscanf(range, "%u-%u", &lo, &hi) == 2 &&
        lo > 0 && lo <= hi && hi <= 65535) {
        lo_port = lo;
        hi_port = hi;
        enabled = 1;
        if (verbose)
            fprintf(stderr, "[port_shim] active, range %u-%u\n", lo_port, hi_port);
    } else if (verbose) {
        fprintf(stderr, "[port_shim] inactive (set PORT_SHIM_RANGE=lo-hi)\n");
    }
}

/* Returns the port for AF_INET/AF_INET6, or -1 if this is not a rewritable
 * address family. Loopback is reported as -1 too: it is never firewalled, and
 * spending the scarce range on it would be wasteful. */
static int port_of(const struct sockaddr *addr)
{
    if (addr->sa_family == AF_INET) {
        const struct sockaddr_in *v4 = (const struct sockaddr_in *)addr;
        if ((ntohl(v4->sin_addr.s_addr) >> 24) == 127)
            return -1;
        return ntohs(v4->sin_port);
    }
    if (addr->sa_family == AF_INET6) {
        const struct sockaddr_in6 *v6 = (const struct sockaddr_in6 *)addr;
        if (IN6_IS_ADDR_LOOPBACK(&v6->sin6_addr))
            return -1;
        return ntohs(v6->sin6_port);
    }
    return -1;
}

static void set_port(struct sockaddr_storage *addr, unsigned port)
{
    if (addr->ss_family == AF_INET)
        ((struct sockaddr_in *)addr)->sin_port = htons(port);
    else
        ((struct sockaddr_in6 *)addr)->sin6_port = htons(port);
}

int bind(int fd, const struct sockaddr *addr, socklen_t len)
{
    bind_fn pass_through = next_bind();
    if (!pass_through) {
        errno = ENOSYS;
        return -1;
    }

    if (!enabled || !addr || len > sizeof(struct sockaddr_storage))
        return pass_through(fd, addr, len);

    if (port_of(addr) != 0)
        return pass_through(fd, addr, len);

    int type = 0;
    socklen_t type_len = sizeof(type);
    if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &type_len) != 0 || type != SOCK_STREAM)
        return pass_through(fd, addr, len);

    /* Start at a pid-derived offset so concurrent ranks on one node do not all
     * contend for the bottom of the range. */
    unsigned span = hi_port - lo_port + 1;
    unsigned offset = (unsigned)getpid() % span;

    struct sockaddr_storage attempt;
    memcpy(&attempt, addr, len);

    for (unsigned i = 0; i < span; i++) {
        unsigned port = lo_port + (offset + i) % span;
        set_port(&attempt, port);
        if (pass_through(fd, (const struct sockaddr *)&attempt, len) == 0) {
            if (verbose)
                fprintf(stderr, "[port_shim] pid %d bound port %u\n", getpid(), port);
            return 0;
        }
        /* Only a port collision is worth retrying; anything else (EACCES,
         * EINVAL, EAFNOSUPPORT) will repeat for every port in the range. */
        if (errno != EADDRINUSE && errno != EADDRNOTAVAIL)
            break;
    }

    if (verbose)
        fprintf(stderr, "[port_shim] pid %d found no free port in %u-%u; using kernel choice\n",
                getpid(), lo_port, hi_port);
    return pass_through(fd, addr, len);
}

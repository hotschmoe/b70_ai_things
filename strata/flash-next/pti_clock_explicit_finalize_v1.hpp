// Default-off output-only metadata for the pinned PTI fixture experiment.
#ifndef B70_PTI_UNITRACE_CLOCK_EXPORT_V1_H
#define B70_PTI_UNITRACE_CLOCK_EXPORT_V1_H
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <locale>
#include <sstream>
#include <string>
#include <utility>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <time.h>
namespace b70_pti_clock_v1 {
inline bool Enabled() {
  const char* value = std::getenv("UNITRACE_CLOCK_METADATA_PATH");
  return value != nullptr && value[0] != '\0';
}
[[noreturn]] inline void Fail(const char* reason) {
  std::fprintf(stderr, "[B70_CLOCK_EXPORT_ERROR] %s\n", reason);
  std::fflush(stderr);
  _exit(86);
}

inline bool BridgeEnabled() {
  const char* flag = std::getenv("UNITRACE_HOST_CLOCK_BRIDGE");
  if (flag == nullptr || flag[0] == '\0' || std::strcmp(flag, "0") == 0) return false;
  if (std::strcmp(flag, "1") != 0 || !Enabled()) Fail("host bridge requires exact1 and metadata output");
  return true;
}
inline uint64_t ReadHostClock(clockid_t clock) {
  timespec value{};
  if (::clock_gettime(clock, &value) != 0 || value.tv_sec < 0 || value.tv_nsec < 0 || value.tv_nsec >= 1000000000L)
    Fail("host bridge CPU clock read failed");
  const uint64_t seconds = static_cast<uint64_t>(value.tv_sec);
  const uint64_t nanoseconds = static_cast<uint64_t>(value.tv_nsec);
  if (seconds > (UINT64_MAX - nanoseconds) / 1000000000ULL) Fail("host bridge CPU clock overflow");
  return seconds * 1000000000ULL + nanoseconds;
}
struct HostBridgeSample { uint64_t raw_before, monotonic, raw_after; };
inline HostBridgeSample ReadHostBridge() {
  HostBridgeSample sample{ReadHostClock(CLOCK_MONOTONIC_RAW), 0, 0};
  sample.monotonic = ReadHostClock(CLOCK_MONOTONIC);
  sample.raw_after = ReadHostClock(CLOCK_MONOTONIC_RAW);
  if (sample.raw_after < sample.raw_before) Fail("host bridge RAW clock reordered");
  return sample;
}
inline uint64_t DoubleBits(double value) {
  static_assert(sizeof(double) == sizeof(uint64_t), "exact F64 payload required");
  uint64_t bits; std::memcpy(&bits, &value, sizeof(bits)); return bits;
}
inline uint64_t UuidWord(const void* value) {
  uint64_t word; std::memcpy(&word, value, sizeof(word)); return word;
}
struct Writer {
  std::mutex lock;
  int fd = -1;
  pid_t owner = 0;
  uint64_t sequence = 0;
  uint64_t bytes = 0;
  bool closed = false;
  bool close_error = false;
  ~Writer() { if (!CloseOnce(false)) Fail("final close failed"); }
  bool CloseOnce(bool require_opened) {
    std::lock_guard<std::mutex> held(lock);
    if (closed) return !close_error;
    if (fd < 0) { closed=true;close_error=require_opened;return !close_error; }
    if (owner != ::getpid()) { close_error=true;return false; }
    if (BridgeEnabled()) {
      const auto sample=ReadHostBridge();
      EmitLocked("host_clock_bridge", {{"site",2},{"raw_before_ns",sample.raw_before},
          {"monotonic_ns",sample.monotonic},{"raw_after_ns",sample.raw_after}});
    }
    const int result=::close(fd);fd=-1;closed=true;close_error=result!=0;
    return !close_error;
  }
  void Emit(const char* kind, std::initializer_list<std::pair<const char*,uint64_t>> fields) {
    // OFF performs no allocation, lock, file operation or extra clock query.
    if (!Enabled()) return;
    std::lock_guard<std::mutex> held(lock);
    if (closed) Fail("metadata emission after explicit close");
    EmitLocked(kind,fields);
  }
  void EmitLocked(const char* kind, std::initializer_list<std::pair<const char*,uint64_t>> fields) {
    const pid_t pid = ::getpid();
    if (fd < 0) {
      const char* supplied = std::getenv("UNITRACE_CLOCK_METADATA_PATH");
      if (supplied == nullptr || std::strcmp(supplied, "/results/unitrace/clock.{pid}.jsonl") != 0)
        Fail("exact declared per-process output template required");
      const std::string path = std::string("/results/unitrace/clock.") + std::to_string(pid) + ".jsonl";
      fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
      if (fd < 0) Fail("exclusive output open failed");
      struct stat info;
      if (::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_nlink != 1 || info.st_uid != ::geteuid())
        Fail("regular owned output required");
      owner = pid;
    }
    if (owner != pid) Fail("inherited writer owner changed");
    std::ostringstream row;
    row.imbue(std::locale::classic());
    row << "{\"schema\":1,\"namespace\":\"b70-pti-clock-export-v1\",\"source_base\":\"6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033\",\"pid\":" << pid
        << ",\"writer_tid\":" << ::syscall(SYS_gettid) << ",\"sequence\":" << ++sequence << ",\"kind\":\"" << kind << "\"";
    for (const auto& field : fields) row << ",\"" << field.first << "\":" << field.second;
    row << "}\n";
    const std::string value = row.str();
    if (value.size() > 4096 || bytes + value.size() > (512u << 20)) Fail("metadata output bound exceeded");
    size_t offset = 0;
    while (offset < value.size()) {
      const ssize_t done = ::write(fd, value.data() + offset, value.size() - offset);
      if (done < 0 && errno == EINTR) continue;
      if (done <= 0) Fail("metadata output write failed");
      offset += static_cast<size_t>(done);
    }
    bytes += value.size();
  }
};
inline Writer& OwnedWriter() { static Writer writer; return writer; }
inline void Emit(const char* kind, std::initializer_list<std::pair<const char*,uint64_t>> fields) {
  if (!Enabled()) return;
  OwnedWriter().Emit(kind,fields);
}
inline bool CloseOnce() {
  if (!Enabled() || !BridgeEnabled()) return false;
  return OwnedWriter().CloseOnce(true);
}

inline void HostBridge(uint64_t site) {
  if (!BridgeEnabled()) return;
  const auto sample = ReadHostBridge();
  Emit("host_clock_bridge", {{"site", site}, {"raw_before_ns", sample.raw_before},
       {"monotonic_ns", sample.monotonic}, {"raw_after_ns", sample.raw_after}});
}
struct QueryBracket { bool enabled; uint64_t raw_before; };
inline QueryBracket BeginQueryBracket() {
  if (!BridgeEnabled()) return {false, 0};
  return {true, ReadHostClock(CLOCK_MONOTONIC_RAW)};
}
inline void EndQueryBracket(const QueryBracket& bracket, uint64_t site, uint64_t device,
                            uint64_t status, uint64_t host_ns, uint64_t device_ticks) {
  if (!bracket.enabled) return;
  // First operation after returning from the original GPU query is this CPU read.
  const uint64_t raw_after = ReadHostClock(CLOCK_MONOTONIC_RAW);
  if (!BridgeEnabled()) Fail("host bridge flag changed during original query");
  if (raw_after < bracket.raw_before) Fail("original timestamp query RAW bracket reordered");
  Emit("global_timestamp_query_bracket", {{"query_site", site}, {"device_handle", device},
       {"return_code", status}, {"returned_pair_present", uint64_t(status == 0)},
       {"host_ns", host_ns}, {"device_ticks", device_ticks},
       {"raw_before_ns", bracket.raw_before}, {"raw_after_ns", raw_after}});
  HostBridge(100 + site);
}
} // namespace b70_pti_clock_v1
#endif
